"""
XML Loader for SemEval-2014 Task 4 dataset.

Reads XML files from data/raw/ and returns a list of RawRecord objects.
Malformed records are logged and skipped — the pipeline never crashes on bad data.

SemEval-2014 XML structure:
    <sentences>
        <sentence id="...">
            <text>The battery life is excellent but delivery was slow.</text>
            <aspectTerms>
                <aspectTerm term="battery life" polarity="positive" from="4" to="16"/>
                <aspectTerm term="delivery" polarity="negative" from="34" to="42"/>
            </aspectTerms>
        </sentence>
    </sentences>
"""

import logging
import re

import xml.etree.ElementTree as ET
from pathlib import Path

try:
    from lxml import etree as lxml_etree
    _LXML_AVAILABLE = True
except ImportError:
    _LXML_AVAILABLE = False

from absa.models import AnnotatedAspect, RawRecord

logger = logging.getLogger(__name__)

# Valid polarity values in SemEval-2014
_VALID_POLARITIES = {"positive", "negative", "neutral", "conflict"}


def load(file_path: str | Path) -> list[RawRecord]:
    """
    Parse a SemEval-2014 XML file and return a list of RawRecord objects.

    Args:
        file_path: Path to the SemEval XML file (e.g. data/raw/Laptops_Train.xml)

    Returns:
        List of RawRecord objects. Malformed records are skipped and logged.

    Raises:
        FileNotFoundError: If the file does not exist.
        ET.ParseError: If the file is not valid XML at the root level.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    try:
        raw_bytes = file_path.read_bytes()
        if _LXML_AVAILABLE:
            # lxml recovery mode handles malformed XML (unescaped quotes, etc.)
            parser = lxml_etree.XMLParser(recover=True, encoding="utf-8")
            lxml_root = lxml_etree.fromstring(raw_bytes, parser=parser)
            # Convert lxml tree to stdlib ET for compatibility
            raw_str = lxml_etree.tostring(lxml_root, encoding="unicode")
            root = ET.fromstring(raw_str)
            tree = ET.ElementTree(root)
        else:
            raw = raw_bytes.decode("utf-8", errors="replace")
            raw = raw.replace("\u201c", '"').replace("\u201d", '"')
            raw = raw.replace("\u2018", "'").replace("\u2019", "'")
            raw = raw.replace("\u2013", "-").replace("\u2014", "-")
            raw = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", raw)
            tree = ET.ElementTree(ET.fromstring(raw))
    except ET.ParseError as e:
        raise ET.ParseError(f"Failed to parse XML file '{file_path}': {e}") from e

    root = tree.getroot()
    records: list[RawRecord] = []
    for sentence_elem in root.findall("sentence"):
        record = _parse_sentence(sentence_elem, file_path.name)
        if record is not None:
            records.append(record)

    logger.info("Loaded %d valid records from '%s'", len(records), file_path.name)
    return records


def _parse_sentence(
    sentence_elem: ET.Element, filename: str
) -> RawRecord | None:
    """Parse a single <sentence> element into a RawRecord. Returns None if malformed."""
    sentence_id = sentence_elem.get("id", "").strip()
    if not sentence_id:
        logger.warning(
            "[%s] Skipping <sentence> with missing or empty 'id' attribute.", filename
        )
        return None

    text_elem = sentence_elem.find("text")
    if text_elem is None or not (text_elem.text or "").strip():
        logger.warning(
            "[%s] Skipping sentence id='%s': missing or empty <text> element.",
            filename, sentence_id,
        )
        return None

    text = text_elem.text.strip()
    aspect_terms = _parse_aspect_terms(sentence_elem, sentence_id, filename)

    return RawRecord(sentence_id=sentence_id, text=text, aspect_terms=aspect_terms)


def _parse_aspect_terms(
    sentence_elem: ET.Element, sentence_id: str, filename: str
) -> list[AnnotatedAspect]:
    """Parse all <aspectTerm> elements within a sentence. Skips malformed entries."""
    aspects: list[AnnotatedAspect] = []
    aspect_terms_elem = sentence_elem.find("aspectTerms")

    if aspect_terms_elem is None:
        # Sentence has no aspect annotations — valid, return empty list
        return aspects

    for at in aspect_terms_elem.findall("aspectTerm"):
        aspect = _parse_single_aspect(at, sentence_id, filename)
        if aspect is not None:
            aspects.append(aspect)

    return aspects


def _parse_single_aspect(
    at_elem: ET.Element, sentence_id: str, filename: str
) -> AnnotatedAspect | None:
    """Parse one <aspectTerm> element. Returns None if any required attribute is missing."""
    term = at_elem.get("term", "").strip()
    polarity = at_elem.get("polarity", "").strip().lower()
    from_str = at_elem.get("from", "")
    to_str = at_elem.get("to", "")

    if not term:
        logger.warning(
            "[%s] sentence id='%s': skipping <aspectTerm> with missing 'term'.",
            filename, sentence_id,
        )
        return None

    if polarity not in _VALID_POLARITIES:
        logger.warning(
            "[%s] sentence id='%s': skipping term='%s' with invalid polarity='%s'.",
            filename, sentence_id, term, polarity,
        )
        return None

    try:
        char_from = int(from_str)
        char_to = int(to_str)
    except (ValueError, TypeError):
        logger.warning(
            "[%s] sentence id='%s': skipping term='%s' with non-integer offsets from='%s' to='%s'.",
            filename, sentence_id, term, from_str, to_str,
        )
        return None

    if char_from < 0 or char_to < 0 or char_from >= char_to:
        logger.warning(
            "[%s] sentence id='%s': skipping term='%s' with invalid offsets from=%d to=%d.",
            filename, sentence_id, term, char_from, char_to,
        )
        return None

    return AnnotatedAspect(
        term=term, polarity=polarity, char_from=char_from, char_to=char_to
    )
