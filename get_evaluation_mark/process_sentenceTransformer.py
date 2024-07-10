from sentence_transformers import SentenceTransformer, util
import argparse
import json

sentences = ["I'm happy", "I'm full of happiness", "I'm sad", "I'm full of sadness"]

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Compute embedding for both lists
embedding_1 = model.encode(sentences[0], convert_to_tensor=True)
embedding_2 = model.encode(sentences[1:], convert_to_tensor=True)

print(util.pytorch_cos_sim(embedding_1, embedding_2))
## tensor([[0.6003]])
