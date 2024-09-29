import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

from .env import GLOVE_PATH

# 1. Load GloVe embeddings
def load_glove_embeddings(glove_file_path, embedding_dim=300):
    glove_embeddings = {}
	# super inefficent and memory consumptive
    with open(glove_file_path, 'r') as f:
        for line in f:
            # Split the line into tokens
            values = line.split()
            
            # The word is everything before the last 'embedding_dim' tokens
            word = ' '.join(values[:-embedding_dim])
            
            # The vector is the last 'embedding_dim' tokens
            vector = np.array(values[-embedding_dim:], dtype='float32')
            glove_embeddings[word] = vector
    return glove_embeddings

# 2. Get word embeddings for a list of words
def get_word_vectors(words, glove_embeddings):
    vectors = []
    for word in words:
        if word in glove_embeddings:
            vectors.append(glove_embeddings[word])
        else:
            vectors.append(np.zeros(300))  # If word not found, use a zero vector
    return np.array(vectors)

# 3. Normalize the word vectors
def normalize_vectors(word_vectors):
    return normalize(word_vectors, axis=1)

# 4. Compute RSA matrix using cosine similarity
def compute_similarity_matrix(word_vectors):
    return cosine_similarity(word_vectors)

# Main function to create RSA matrix
def create_rsa_matrix(words, glove_file_path=GLOVE_PATH):
    glove_embeddings = load_glove_embeddings(glove_file_path)
    word_vectors = get_word_vectors(words, glove_embeddings)
    
    # Normalize word vectors before computing cosine similarity
    normalized_vectors = normalize_vectors(word_vectors)
    
    # Compute similarity matrix
    similarity_matrix = compute_similarity_matrix(normalized_vectors)
    
    return similarity_matrix

