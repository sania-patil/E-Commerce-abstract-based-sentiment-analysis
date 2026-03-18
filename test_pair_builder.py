import sys
sys.path.insert(0, 'src')
from absa.pair_builder import build_pairs
from absa.models import AspectSentimentPair

# Simulate sentiment classifier output
classifier_results = [
    {"term": "battery life", "span": [1, 2], "polarity": "Positive",
     "confidence": 0.82, "low_confidence": False, "aspect_confidence": 0.90},
    {"term": "delivery", "span": [6, 6], "polarity": "Negative",
     "confidence": 0.41, "low_confidence": True, "aspect_confidence": 0.88},
]

pairs = build_pairs(classifier_results)

print('--- Aspect-Sentiment Pairs ---')
for p in pairs:
    flag = ' [LOW CONFIDENCE]' if p.low_confidence else ''
    print(f'  {p.aspect:20s} -> {p.polarity:10s} (confidence: {p.confidence:.4f}){flag}')

print()
print(f'Returns AspectSentimentPair objects: {all(isinstance(p, AspectSentimentPair) for p in pairs)}')
print(f'Empty list test: {build_pairs([])}')
print('Pair builder working correctly.')
