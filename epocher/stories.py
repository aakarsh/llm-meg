from .env import * 
import os
import re
import nltk
from collections import Counter

# Ensure you have the necessary NLTK data
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

task_stimuli = ['lw1', 'cable_spool_fort','easy_money','the_black_willow' ]

def load_stimuli(directory):
    content_map = {}
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            with open(os.path.join(directory, filename), 'r') as file:
                content = file.read()
                # Assuming you want to store the content by filename
                content_map[filename] = content
    return content_map


def select_words_by_part_of_speech(words, 
        num_words=None, word_pos=['VB']):
    """
    Filter the words by the parts of speech
    """
    tagged_words = nltk.pos_tag(words)

    def is_selectable_tag(current_tag, tags=['VB']):
        for filter_tag in tags: 
            if current_tag.startswith(filter_tag):
                return True
        return False

    selected_words = [word for word, tag in tagged_words if len(word)>2 and 
            is_selectable_tag(tag, tags=word_pos) ]
    return list(set(selected_words))

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

STORY_CHACHE = None

def load_experiment_stories(): 
    global STORY_CHACHE
    if not STORY_CHACHE:
       STORY_CHACHE = load_stimuli(MEG_MASC_STIMULI_DIRECTORY)
    return  STORY_CHACHE

def load_experiment_salient_words():
    stories_map = load_experiment_stories()
    return get_salient_words(stories_map[list(stories_map.keys())[0]])

def get_story_key(task_id):
    story_key = f'{task_stimuli[task_id]}.txt'
    return story_key 

def get_experiment_story(task_id):
    story_map = load_experiment_stories()
    story_key = get_story_key(task_id)
    story = story_map[story_key]
    return story
