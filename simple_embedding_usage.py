
# !pip install sentence-transformers
# !pip install numpy

from sentence_transformers import SentenceTransformer, util

# Load the GTE-large model
model = SentenceTransformer("thenlper/gte-large",token='')

# Your list of candidate sentences
sentences = [
    "That is a happy person",
    "That is a happy dog",
    "That is a very happy person",
    "Today is a sunny day"
]

# Pre-compute the embeddings once
sentence_embeddings = model.encode(sentences, convert_to_tensor=True)

def get_best_match(query: str, sentences: list, sentence_embeddings):
    # Encode the query sentence
    query_embedding = model.encode(query, convert_to_tensor=True)
    
    # Compute cosine similarities
    similarity_scores = util.pytorch_cos_sim(query_embedding, sentence_embeddings)[0]
    
    # Get the index of the highest score
    best_index = int(similarity_scores.argmax())
    best_score = float(similarity_scores[best_index])
    best_sentence = sentences[best_index]
    
    return best_sentence, best_score

# Example use
query_sentence = "Sun is too hot"
match, score = get_best_match(query_sentence, sentences, sentence_embeddings)

print(f"Query       : {query_sentence}")
print(f"Best match  : {match}")
print(f"Similarity  : {score:.4f}")
