import torch
import numpy as np

from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from nltk.tokenize import sent_tokenize

from .env import GLOVE_PATH
from epocher.stories import load_experiment_stories
from collections import defaultdict

CACHED_EMBEDDING = {}

# Initialize tokenizer and model
 
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', clean_up_tokenization_spaces=False)
model = BertModel.from_pretrained('bert-base-uncased')

task_stimuli = ['lw1', 'cable_spool_fort','easy_money','the_black_widow' ]

EMBEDDING_TASK_CACHE = {}

def get_index_stimulus_stories(word_index, task_id, use_cache=True):
    """
    get_index_stimulus_stories - story_path
    """
    story_key = f'{task_stimuli[task_id]}.txt'
    """
    if story_key in EMBEDDING_TASK_CACHE:
        return  EMBEDDING_TASK_CACHE[story_key]
    """

    story_map = load_experiment_stories()
    story = story_map[story_key]

    # Initialize dictionaries
    word_embeddings = defaultdict(lambda: np.zeros((model.config.hidden_size,)))
    word_counts = defaultdict(int)
    
    # Step 1: Split the story into sentences
    sentences = sent_tokenize(story)
    for idx, sentence in enumerate(sentences):
        inputs = tokenizer(sentence, return_tensors='pt', truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            # shape: (batch_size, sequence_length, hidden_size)
            embeddings = outputs[0]  

            for token, embedding in zip(inputs['input_ids'][0], embeddings[0]):
                    word = tokenizer.decode([token.item()])
                    if word.strip() and word not in ['[CLS]', '[SEP]']:
                        word_embeddings[word] += embedding.numpy()
                        word_counts[word] += 1

    # Average the embeddings
    average_embeddings = {word: word_embeddings[word] / word_counts[word] 
                                  for word in word_embeddings if word_counts[word] > 0}

    avg_word_embeddings = []
    word_index = []
    for word_to_avg in word_index:
        word_inputs = tokenizer(word_to_avg, return_tensors='pt', truncation=True, padding=True)
        decoded_word = tokenizer.decode(word_inputs['input_ids'][0])
        word_stem = decoded_word.split()[1]
        print(f"word_stem : {word_stem}")
        if not word_stem in average_embeddings:
            print(f'skipping {word_to_avg}, {word_stem} not found ')
            continue
        else:
            print(f'found {word_to_avg}')
        word_index.append(word_to_avg)
        avg_word_embedding =  average_embeddings[word_stem]
        avg_word_embeddings.append(avg_word_embedding)


    retval_embeddings = torch.empty((0, 768))  # Handle case when no embeddings are found
    if avg_word_embeddings:  # Check if list is not empty
        retval_embeddings = torch.cat([torch.tensor(embedding).unsqueeze(0) 
                                        for embedding in avg_word_embeddings])
    
    return word_index, retval_embeddings 


# 3. Normalize the word vectors
def normalize_vectors(word_vectors):
    return normalize(word_vectors, axis=1)

# 4. Compute RSA matrix using cosine similarity
def compute_similarity_matrix(word_vectors):
    return cosine_similarity(word_vectors)



def create_rsa_matrix(words, task_id):
    word_vectors = get_index_stimulus_stories(words, task_id)
    print(f"len word vectors {len(word_vectors)}")
    # Normalize word vectors before computing cosine similarity
    normalized_vectors = normalize_vectors(word_vectors)
    
    # Compute similarity matrix
    similarity_matrix = compute_similarity_matrix(normalized_vectors)
    
    return similarity_matrix

