import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

from .env import GLOVE_PATH

# Hash of word embeddings if it has been fetched, alternative to 
# loading and searching through glove embedding file.
CACHED_EMBEDDING = {}

def load_glove_embeddings(glove_file_path, embedding_dim=300, use_cache=True):
    """
    Load glove embeddings using the provide glove embedding filepath.
    """
    global CACHED_EMBEDDING  # Declare CACHED_EMBEDDING as global

    if use_cache and CACHED_EMBEDDING:
        return CACHED_EMBEDDING

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

    CACHED_EMBEDDING = glove_embeddings
    return glove_embeddings

def get_word_vectors(words, glove_embeddings):
    """
    Convert map of glove embeddings into a list of vectors of following the same 
    ordering as provided list of words.
    """
    vectors = []
    for word in words:
        if word in glove_embeddings:
            vectors.append(glove_embeddings[word])
        else:
            vectors.append(np.zeros(300))  # If word not found, use a zero vector
    return np.array(vectors)

def normalize_vectors(word_vectors):
    """
    Normalize word vectors to be used a pre-processing for 
    cosine similarity computation.
    """
    return normalize(word_vectors, axis=1)

def compute_similarity_matrix(word_vectors):
    """
    Compute the cosine similarity for word vectors.
    """
    return cosine_similarity(word_vectors)

def create_rsa_matrix(words, glove_file_path=GLOVE_PATH):
    """
    Construct a RSA matrix to compare stimulus similarity 
    accross model representations.
    """
    glove_embeddings = load_glove_embeddings(glove_file_path)
    word_vectors = get_word_vectors(words, glove_embeddings)
    
    # Normalize word vectors before computing cosine similarity
    normalized_vectors = normalize_vectors(word_vectors)
    
    # Compute similarity matrix
    similarity_matrix = compute_similarity_matrix(normalized_vectors)
    
    return similarity_matrix


