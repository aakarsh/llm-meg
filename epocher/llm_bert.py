import torch
import numpy as np

from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from nltk.tokenize import sent_tokenize

from .env import GLOVE_PATH
from epocher.stories import load_experiment_stories
from collections import defaultdict


import hashlib
import pickle

# Initialize tokenizer and model
from transformers import BertTokenizerFast
 
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased', clean_up_tokenization_spaces=False)
model = BertModel.from_pretrained('bert-base-uncased')
#'cable_spool_fort.txt', 'the_black_willow.txt', 'easy_money.txt', 'lw1.txt'
task_stimuli = ['lw1', 'cable_spool_fort','easy_money','the_black_willow' ]

EMBEDDING_TASK_CACHE = {}



def compute_md5_hash(word_index, task_id):
    # Combine word_index and task_id into a string and compute MD5
    data_to_hash = f"{task_id}_{'_'.join(word_index)}"
    return hashlib.md5(data_to_hash.encode()).hexdigest()

def cache_result(hash_key, result):
    # Store the result in the cache using the hash_key
    EMBEDDING_TASK_CACHE[hash_key] = result

def get_cached_result(hash_key):
    # Retrieve from the cache
    return EMBEDDING_TASK_CACHE.get(hash_key, None)

def get_whole_word_embeddings(word_index, task_id, use_cache=True):
    story_key = f'{task_stimuli[task_id]}.txt'

    # Compute the hash of the inputs
    hash_key = compute_md5_hash(word_index, task_id)
    
    # Check if the cache contains a result for this hash
    if use_cache and hash_key in EMBEDDING_TASK_CACHE:
        print(f"Loading from cache: {hash_key}")
        return EMBEDDING_TASK_CACHE[hash_key]


    story_map = load_experiment_stories()
    story = story_map[story_key]

    # Initialize dictionaries
    # Step 1: Split the story into sentences
    words_found = []
    embeddings_found = []

    word_embeddings = defaultdict(lambda: np.zeros((model.config.hidden_size,)))
    word_counts = defaultdict(int)

    sentences = sent_tokenize(story)
    
    for idx, sentence in enumerate(sentences):
        inputs = tokenizer(sentence, return_tensors='pt', add_special_tokens=True, return_offsets_mapping=True)

        offset_mapping = inputs['offset_mapping'][0]  # This maps tokens back to character-level offsets in the original text
        inputs.pop('offset_mapping')

        with torch.no_grad():
            outputs = model(**inputs)
            # shape: (batch_size, sequence_length, hidden_size)
            token_embeddings = outputs.last_hidden_state  # Shape: (batch_size, sequence_length, hidden_size)

        # Decode tokenized words
        tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        # Store aggregated embeddings for each word
        whole_word_embeddings = []
        current_word = ""
        current_word_embedding = []
        current_word_start = None

        for idx, token in enumerate(tokens):
            # Skip [CLS] and [SEP] tokens
            if token in ['[CLS]', '[SEP]']:
                continue

            # If token is part of a new word (no '##' prefix), process previous word
            if not token.startswith('##'):
                if current_word:  # Save previous word embedding
                    current_word_embedding = torch.mean(torch.stack(current_word_embedding), dim=0)
                    whole_word_embeddings.append(current_word_embedding)
                    words_found.append(current_word)
                    embeddings_found.append(current_word_embedding)
                    word_embeddings[current_word] += current_word_embedding.numpy()
                    word_counts[current_word] += 1

                current_word = token
                current_word_embedding = [token_embeddings[0, idx]]
                current_word_start = offset_mapping[idx][0]
            else:  # This is a subword, add its embedding to the current word
                current_word += token[2:]  # Remove '##'
                current_word_embedding.append(token_embeddings[0, idx])


        # Add the last word
        if current_word:
            current_word_embedding  = torch.mean(torch.stack(current_word_embedding), dim=0)
            whole_word_embeddings.append(current_word_embedding)
            words_found.append(current_word)
            embeddings_found.append(current_word_embedding)
            word_embeddings[current_word] += current_word_embedding.numpy()
            word_counts[current_word] += 1

    # Average the embeddings
    average_embeddings = { word: word_embeddings[word] / word_counts[word] 
                                  for word in word_embeddings if word_counts[word] > 0 }

    found_words_in_index = []
    found_words_embeddings = []

    for word in word_index:
        lower_case_word = word.lower()
        if lower_case_word in average_embeddings:
            avg_word_embedding =  average_embeddings[word.lower()]
            found_words_embeddings.append(avg_word_embedding) 
            found_words_in_index.append(word)
            print("found word adding to index", word)


    # Cache the result
    
    result = (found_words_in_index, { word: found_words_embeddings[idx] for idx, word in enumerate(found_words_in_index)})
    cache_result(hash_key, result)
    return result 


# 3. Normalize the word vectors
def normalize_vectors(word_vectors):
    return normalize(word_vectors, axis=1)

# 4. Compute RSA matrix using cosine similarity
def compute_similarity_matrix(word_vectors):
    return cosine_similarity(word_vectors)


# 2. Get word embeddings for a list of words
def get_word_vectors(words, glove_embeddings):
    vectors = []
    for word in words:
        if word in glove_embeddings:
            vectors.append(glove_embeddings[word])
        else:
            vectors.append(np.zeros(300))  # If word not found, use a zero vector
    return np.array(vectors)


def create_rsa_matrix(words, task_id):
    words_found, word_embeddings = get_whole_word_embeddings(words, task_id)
    word_vectors = get_word_vectors(words_found, word_embeddings)
    print(f"len word vectors {len(word_vectors)}, for words: {len(words)} found words {len(words_found)}")
    # Normalize word vectors before computing cosine similarity
    normalized_vectors = normalize_vectors(word_vectors)
    
    # Compute similarity matrix
    similarity_matrix = compute_similarity_matrix(normalized_vectors)
    
    return words_found, similarity_matrix

