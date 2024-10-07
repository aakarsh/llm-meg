from epocher.stories import *
from epocher.env import *
import stanza
from nltk.tokenize import sent_tokenize

# Tense, Aspect, Mood, Person, and Number are usually considered morphological,
nlp = stanza.Pipeline('en', processors='tokenize,mwt,pos,lemma,depparse')

# Function to process a single story file and extract verbs with features
def extract_verbs_from_sentence(sentence):
    doc = nlp(sentence)
    verbs_data = []
    for sent in doc.sentences:
        for word in sent.words:
            if word.upos == 'VERB':  # Check if the word is a verb
                verb_info = {
                    'verb': word.text,
                    'lemma': word.lemma,
                    'tense': word.feats.get('Tense', 'N/A'),
                    'aspect': word.feats.get('Aspect', 'N/A'),
                    'voice': word.feats.get('Voice', 'N/A'),
                    'mood': word.feats.get('Mood', 'N/A'),
                    'person': word.feats.get('Person', 'N/A'),
                    'number': word.feats.get('Number', 'N/A'),
                    'other_features': word.feats
                }
                verbs_data.append(verb_info)
    return verbs_data

def mark_verb_features(task_id, verb_list):
    verb_set = set(verb_list)
    story = get_experiment_story(task_id)
    sentences = sent_tokenize(story)
    verb_features = {}

    for idx, sentence in enumerate(sentences):
        doc = nlp(sentence)
        for sent in doc.sentences:
            for word in sent.words:
                if word.upos == 'VERB' and word.text.lower() in verb_set:  
                    features = {feat.split('=')[0]: feat.split('=')[1] for feat in word.feats.split('|') if '=' in feat}

                    if word.text in verb_features:
                        print(f'duplicate verb {word.text}')
                    else:
                        verb_features[word.text.lower()] = {
                            'verb': word.text,
                            'lemma': word.lemma,
                            'tense': features.get('Tense', 'N/A'),
                            'aspect': features.get('Aspect', 'N/A'),
                            'voice': features.get('Voice', 'N/A'),
                            'mood': features.get('Mood', 'N/A'),
                            'person': features.get('Person', 'N/A'),
                            'number': features.get('Number', 'N/A'),
                            "other": word.feats
                        }
    return verb_features
