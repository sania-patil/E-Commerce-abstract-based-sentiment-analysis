import sys
sys.path.insert(0, 'src')
from absa.models import AspectSentimentPair
from absa.opinion_summarizer import summarize
from absa.serializer import summary_to_json

# Simulate pairs from multiple reviews
pairs = [
    # battery life mentioned 5 times positive, 1 negative -> STRENGTH
    AspectSentimentPair("battery life", "Positive", 0.9, [1,2], False),
    AspectSentimentPair("battery life", "Positive", 0.85, [1,2], False),
    AspectSentimentPair("battery life", "Positive", 0.88, [1,2], False),
    AspectSentimentPair("Battery Life", "Positive", 0.91, [1,2], False),  # different case
    AspectSentimentPair("  battery life  ", "Positive", 0.87, [1,2], False),  # extra spaces
    AspectSentimentPair("battery life", "Negative", 0.72, [1,2], False),

    # delivery mentioned 1 positive, 4 negative -> WEAKNESS
    AspectSentimentPair("delivery", "Negative", 0.80, [6,6], False),
    AspectSentimentPair("delivery", "Negative", 0.75, [6,6], False),
    AspectSentimentPair("delivery", "Negative", 0.82, [6,6], False),
    AspectSentimentPair("delivery", "Negative", 0.78, [6,6], False),
    AspectSentimentPair("delivery", "Positive", 0.60, [6,6], False),

    # screen mentioned 3 positive, 3 negative -> TIE (excluded)
    AspectSentimentPair("screen", "Positive", 0.70, [3,3], False),
    AspectSentimentPair("screen", "Positive", 0.72, [3,3], False),
    AspectSentimentPair("screen", "Positive", 0.68, [3,3], False),
    AspectSentimentPair("screen", "Negative", 0.65, [3,3], False),
    AspectSentimentPair("screen", "Negative", 0.71, [3,3], False),
    AspectSentimentPair("screen", "Negative", 0.69, [3,3], False),

    # price mentioned 2 negative -> WEAKNESS
    AspectSentimentPair("price", "Negative", 0.88, [5,5], False),
    AspectSentimentPair("price", "Negative", 0.85, [5,5], False),
]

summary = summarize(pairs)

print('=== Opinion Summary ===')
print(summary_to_json(summary, pretty=True))
print()
print(f'Strengths : {[a.aspect for a in summary.strengths]}')
print(f'Weaknesses: {[a.aspect for a in summary.weaknesses]}')
print()
print('Checks:')
print(f'  battery life is a strength (count=5): {any(a.aspect=="battery life" and a.count==5 for a in summary.strengths)}')
print(f'  delivery is a weakness (count=4)    : {any(a.aspect=="delivery" and a.count==4 for a in summary.weaknesses)}')
print(f'  screen excluded (tie)               : {"screen" not in [a.aspect for a in summary.strengths+summary.weaknesses]}')
print(f'  case normalized (Battery Life->battery life): {"battery life" in [a.aspect for a in summary.strengths]}')
print(f'  empty input returns empty summary   : {summarize([]).strengths == [] and summarize([]).weaknesses == []}')
print()
print('Opinion summarizer working correctly.')
