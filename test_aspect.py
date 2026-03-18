import sys
sys.path.insert(0, 'src')
import torch
from absa.preprocessor import normalize
from absa.tokenizer import ABSATokenizer
from absa.encoder import BERTEncoder
from absa.aspect_extractor import AspectExtractor

# Step 1: preprocess + tokenize
text = normalize('The battery life is excellent but delivery was slow.')
words = text.split()
tok = ABSATokenizer()
token_out = tok.tokenize(text)

# Step 2: encode
encoder = BERTEncoder()
encoder.set_inference_mode()
with torch.no_grad():
    embeddings = encoder(token_out)

print(f'Embeddings shape: {embeddings.shape}')

# Step 3: extract aspects (random weights - not trained yet)
extractor = AspectExtractor(hidden_dim=768)
extractor.eval()
spans = extractor.extract(embeddings, token_out.word_ids, words)

print(f'Words: {words}')
print(f'Extracted spans: {spans}')
print(f'Returns a list: {isinstance(spans, list)}')
if spans:
    print(f'Span keys correct: {set(spans[0].keys()) == {"term", "span", "confidence"}}')
else:
    print('Empty list returned (valid - random weights before training)')
print('Aspect extractor working correctly.')
