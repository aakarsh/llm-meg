
import json
import numpy as np
import mne
from mne_bids import BIDSPath # Import the BIDSPath class

# Create a BIDSPath object


DATASET_ROOT="/content/drive/MyDrive/TUE-SUMMER-2024/ulm-meg/"

def load_bids_path(root=DATASET_ROOT, subject="01", datatype="meg", session="0",  task="1"):
    bids_path = BIDSPath(root=DATASET_ROOT, subject=subject, session=session, task=task,datatype=datatype)
    return bids_path

def find_word_events(raw):
    pass

def find_sentence_events(raw):
    pass

def is_word(event_json):
    return event_json['kind'] == "word"

def is_phoneme(event_json):
    return event_json['kind'] == 'phoneme'

def filter_word_events(event_json_strings):
    event_jsons = list(map(parse_event_description, event_json_strings))
    return list(filter(is_word, event_jsons))

def find_word_events_from_annotation(raw):
    all_events, all_event_id = mne.events_from_annotations(raw)
    word_events = list(map(str, filter_word_events(all_event_id)))
    all_word_events = []
    all_word_event_id = {}
    for word_event in word_events:
        word_event_id = all_event_id[word_event]
        for event in all_events:
            if word_event_id == event[2]:
                all_word_events.append(event)
                all_word_event_id[word_event] = word_event_id

    return all_word_events, all_word_event_id

def create_word_epochs(raw, tmin=-.01, tmax=.05):
   all_events, all_event_id = dataset.find_word_events_from_annotation(raw) 
   epochs = mne.Epochs(raw, event_id = all_event_id, detrend=1, baseline=None, event_repeated='drop', tmin=tmin,tmax=tmax)
   return epochs

def parse_event_description(event_json_str):
    event_json_str = event_json_str.replace('\'', '"')
    return json.loads(event_json_str)
