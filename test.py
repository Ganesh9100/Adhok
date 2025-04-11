# installation
# !pip install langchain
# !pip install langchain_community
# !pip install -U langchain-ollama


from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

embedder = FastEmbedEmbeddings(model_name="thenlper/gte-large")

# List of sentences
sentences = [
    "The cat is sitting on the mat.",
    "A dog is playing in the garden.",
    "The quick brown fox jumps over the lazy dog.",
    "A person is riding a bicycle.",
    "It is raining outside."
]

# Generate embeddings
sentence_embeddings = list(embedder.embed(sentences, batch_size=5))



# Sentence to search
query = "An animal jumping over something."

query_embedding = list(embedder.embed([query]))[0]

# Calculate cosine similarity
similarities = cosine_similarity([query_embedding], sentence_embeddings)[0]

# Find the index of the best match
best_index = int(np.argmax(similarities))
best_sentence = sentences[best_index]

print(f"Query: {query}")
print(f"Best match: {best_sentence}")
print(f"Similarity score: {similarities[best_index]:.4f}")
