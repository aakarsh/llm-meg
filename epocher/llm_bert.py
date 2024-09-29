import torch
import numpy as np

from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

from .env import GLOVE_PATH
form .stories import load_experiment_stories

CACHED_EMBEDDING = {}

# Initialize tokenizer and model
 
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', clean_up_tokenization_spaces=False)
model = BertModel.from_pretrained('bert-base-uncased')

task_stimuli = ['lw1', 'cable_spool_fort','easy_money','the_black_widow' ]

STORY_CACHE = None

def get_index_stimulus_stories(word_index, task_id):
    """
    get_index_stimulus_stories - story_path
    """
    story_key = f'{task_stimuli[task_id]}.txt'
    story = load_experiment_salient_words(story_key)


    # tokenize the text
    inputs = tokenizer(story, return_tensors="pt", 
                        padding=True, truncation=True)

    input_ids = inputs['input_ids']

    # Get embeddings from BERT
    with torch.no_grad():
        outputs = model(input_ids)
        embeddings = outputs.last_hidden_state  # Shape: [batch_size, sequence_length, hidden_size]


    # Decode the tokenized inputs to get the token-ids and find positions of the word 'cat'
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    for word_to_avg in word_index:
        word_positions = [i for i, token in enumerate(tokens) 
                                            if token == word_to_avg]

        # Extract the embeddings for the occurrences of the word 'cat'
        word_embeddings = embeddings[0, word_positions, :]  # Extract embeddings for the specific positions

        # Compute the average embedding
        avg_embedding = word_embeddings.mean(dim=0)

        print(f"Average embedding for '{word_to_avg}': {avg_embedding}")

    return np.array([avg_embedding])


def create_contextual_embeddings():
    pass

def create_rsa_matrix(words):
    pass

