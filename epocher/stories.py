from .env import * 
import os
import re
import nltk
from collections import Counter

# Ensure you have the necessary NLTK data
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

def load_stimuli(directory):
    content_map = {}
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            with open(os.path.join(directory, filename), 'r') as file:
                content = file.read()
                # Assuming you want to store the content by filename
                content_map[filename] = content
    return content_map



def get_salient_words(all_text, num_words=20):
    # Tokenization: Remove punctuation and split into words
    words = re.findall(r'\b\w+\b', all_text.lower())
    
    # POS tagging
    tagged_words = nltk.pos_tag(words)
    
    # Filter for nouns and verbs
    salient_words = [word for word, tag in tagged_words if len(word)>2 and tag.startswith('NN') ] #or tag.startswith('VB')]
    
    # Count frequencies
    word_counts = Counter(salient_words)
    
    # Get the most common words
    most_common = word_counts.most_common(num_words)
    
    return most_common

def load_experiment_stories(): 
	return  load_stimuli(MEG_MASC_STIMULI_DIRECTORY)

def load_experiment_salient_words():
	stories_map = load_experiment_stories()
	return get_salient_words(stories_map[list(stories_map.keys())[0]])

def parse_stories():
    pass
