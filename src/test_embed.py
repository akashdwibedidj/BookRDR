from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')


sentences = ['Habits are powerful', 'Habits are easy to change', 'Investment is important', 'Investment is risky']

embeddings = model.encode(sentences)

print(embeddings)
print(len(embeddings))       # number of sentences
print(len(embeddings[0]))    # embedding dimension