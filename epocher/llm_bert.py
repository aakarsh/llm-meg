import torch
import numpy as np

from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from nltk.tokenize import sent_tokenize

from .env import GLOVE_PATH
from epocher.stories import load_experiment_stories

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
    # tokenize the text
    inputs = tokenizer(story, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs['input_ids']

    # Get embeddings from BERT
    with torch.no_grad():
        outputs = model(input_ids)
        embeddings = outputs.last_hidden_state  # Shape: [batch_size, sequence_length, hidden_size]

    # Decode the tokenized inputs to get the token-ids and find positions of the word 'cat'
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    print(tokens)
    for word_to_avg in word_index:
        word_positions = [ i for i, token in enumerate(tokens) 
                                            if token == word_to_avg]

        if len(word_positions) == 0:
            raise RuntimeError(f'Not found: {word_to_avg}')
        print(word_positions)
        # Extract the embeddings for the occurrences of the word 'cat'
        word_embeddings = embeddings[0, word_positions, :]  # Extract embeddings for the specific positions

        print("word_embedding", word_embeddings)
        # Compute the average embedding
        avg_embedding = word_embeddings.mean(dim=0)

        print(f"Average embedding for '{word_to_avg}': {avg_embedding}")

    return np.array([avg_embedding])


def create_contextual_embeddings():
    pass

def create_rsa_matrix(words):
    pass

