import sys
sys.path.insert(0, 'src')
from absa.models import AspectSentimentPair, AspectCount, OpinionSummary
from absa.serializer import pairs_to_json, pairs_from_json, summary_to_json, summary_from_json

# Test pairs serialization
pairs = [
    AspectSentimentPair(aspect="battery life", polarity="Positive", confidence=0.82, span=[1,2], low_confidence=False),
    AspectSentimentPair(aspect="delivery", polarity="Negative", confidence=0.41, span=[6,6], low_confidence=True),
]

print('--- Compact JSON ---')
compact = pairs_to_json(pairs)
print(compact)

print()
print('--- Pretty JSON ---')
pretty = pairs_to_json(pairs, pretty=True)
print(pretty)

print()
print('--- Round-trip test ---')
restored = pairs_from_json(compact)
re_serialized = pairs_to_json(restored)
print(f'Round-trip identical: {compact == re_serialized}')

print()
print('--- Opinion Summary JSON ---')
summary = OpinionSummary(
    strengths=[AspectCount(aspect="battery life", count=45), AspectCount(aspect="screen", count=30)],
    weaknesses=[AspectCount(aspect="delivery", count=23), AspectCount(aspect="price", count=15)],
)
summary_json = summary_to_json(summary, pretty=True)
print(summary_json)

print()
restored_summary = summary_from_json(summary_json)
print(f'Summary round-trip: strengths={[a.aspect for a in restored_summary.strengths]}')
print(f'                    weaknesses={[a.aspect for a in restored_summary.weaknesses]}')
print('Serializer working correctly.')
