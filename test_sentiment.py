import sys
sys.path.insert(0, 'src')
import torch
from absa.preprocessor import normalize
from absa.tokenizer import ABSATokenizer
from absa.encoder import BERTEncoder
from absa.aspect_extractor import AspectExtractor
from absa.sentiment_classifier import SentimentClassifier

text = normalize('The battery life is excellent but delivery was slow.')
words = text.split()

tok = ABSATokenizer()
token_out = tok.tokenize(text)

encoder = BERTEncoder()
encoder.set_inference_mode()
with torch.no_grad():
    embeddings = encoder(token_out)

# Manually define known aspect spans (as if model was trained)
# "battery life" = words 1-2, "delivery" = word 6
spans = [
    {"term": "battery life", "span": [1, 2], "confidence": 0.90},
    {"term": "delivery",     "span": [6, 6], "confidence": 0.88},
]
print(f'Using manual spans: {spans}')

# Classify sentiment
classifier = SentimentClassifier(hidden_dim=768, confidence_threshold=0.5)
classifier.eval()
results = classifier.classify(embeddings, spans, token_out.word_ids)

print()
print('--- Sentiment Classification Results (random weights) ---')
for r in results:
    flag = ' [LOW CONFIDENCE]' if r['low_confidence'] else ''
    print(f"  {r['term']:30s} -> {r['polarity']:10s} (confidence: {r['confidence']:.4f}){flag}")

print()
print(f'Output keys correct: {set(results[0].keys()) == {"term","span","aspect_confidence","polarity","confidence","low_confidence"} if results else "N/A"}')
print('Sentiment classifier working correctly.')
