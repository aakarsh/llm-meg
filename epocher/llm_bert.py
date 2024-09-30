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
from transformers import BertTokenizerFast
 
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased', clean_up_tokenization_spaces=False)
model = BertModel.from_pretrained('bert-base-uncased')

task_stimuli = ['lw1', 'cable_spool_fort','easy_money','the_black_widow' ]

EMBEDDING_TASK_CACHE = {}


def get_whole_word_embeddings(word_index, task_id):
    story_key = f'{task_stimuli[task_id]}.txt'
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
        text = sentence
        inputs = tokenizer(text, return_tensors='pt', add_special_tokens=True, return_offsets_mapping=True)

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


                    print("found word", current_word)
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
            print("found word", current_word)


    print("compute avarege_word embeddings")
    # Average the embeddings
    average_embeddings = { word: word_embeddings[word] / word_counts[word] 
                                  for word in word_embeddings if word_counts[word] > 0}

    found_words_in_index = []
    fournd_words_embeddings = []
    retval_embeddings = torch.empty((0, 768)) 
    for word in word_index:
        lower_case_word = word.lower()
        if lower_case_word in average_embeddings:
            avg_word_embedding =  average_embeddings[word.lower()]
            found_words_embeddings.append(avg_word_embedding) 
        
        
    return whole_word_embeddings




def get_index_stimulus_stories(word_index, task_id, use_cache=True):
    """
    get_index_stimulus_stories - story_path
    """
    """
    if story_key in EMBEDDING_TASK_CACHE:
        return  EMBEDDING_TASK_CACHE[story_key]
    """
    story_key = f'{task_stimuli[task_id]}.txt'
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

