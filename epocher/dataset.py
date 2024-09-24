import pandas as pd
import json
import numpy as np
import mne
from mne_bids import BIDSPath # Import the BIDSPath class
import mne
import mne_bids
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, scale
from tqdm import trange
from wordfreq import zipf_frequency
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import matplotlib

from .env import *
from . import stories as S 

# Create a BIDSPath object
DATASET_ROOT="/content/drive/MyDrive/TUE-SUMMER-2024/ulm-meg/"

ph_info = pd.read_csv(MEG_MASC_ROOT+"/phoneme_info.csv")


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
   all_events, all_event_id = find_word_events_from_annotation(raw) 
   epochs = mne.Epochs(raw, event_id = all_event_id, detrend=1, baseline=None, event_repeated='drop', tmin=tmin,tmax=tmax)
   return epochs

def parse_event_description(event_json_str):
    event_json_str = event_json_str.replace('\'', '"')
    return json.loads(event_json_str)


def create_rsa_matrix():
    print("How do I create a representational similairty matrix.")
    pass

def get_layer_activations():
    print("How do i get layer activations")
    pass


def load_subject_information():
    # Read information about subjects
    subjects = pd.read_csv(MEG_MASC_ROOT +  "/participants.tsv", sep="\t")
    return subjects

def load_subject_ids():
    # find subject id
    subjects = load_subject_information()
    subjects = subjects.participant_id.apply(lambda x: x.split("-")[1]).values
    return subjects 

def _load_raw_meta(raw):
    # preproc annotations
    meta = list()
    for annot in raw.annotations:
        d = eval(annot.pop("description")) # direct of the annotation
        for k, v in annot.items():
            assert k not in d.keys()
            d[k] = v
        meta.append(d)
    meta = pd.DataFrame(meta)
    meta["intercept"] = 1.0
    return meta


# [ ] Need to segment by word instead!
def segment_by_word(raw, tmax=0.9):
    """
    Loads the segmeted word data
    """
    # preprocess annotations.
    meta = _load_raw_meta(raw)
    meta = meta.query('kind=="word"').copy()
    word_events = np.c_[meta.onset*raw.info['sfreq'], 
                                np.ones((len(meta), 2))].astype(int)
    word_epochs = mne.Epochs(raw,
                                word_events, 
                                tmin=-.2, 
                                tmax=tmax, # TODO: this maynot be right.
                                decim=10, 
                                baseline=(-0.2,0.0), 
                                metadata=meta, 
                                preload=True, 
                                event_repeated="drop")
    word_epochs = _threshold_baseline_epochs(word_epochs)
    return word_epochs


def _threshold_baseline_epochs(epochs, threshold_percentile=95):
    """
    Threshold and baseline the epochs of 
    """
    # threshold
    th = np.percentile(np.abs(epochs._data), threshold_percentile)
    # Treshold the data ?
    epochs._data[:] = np.clip(epochs._data, -th, th)
    epochs.apply_baseline()
    th = np.percentile(np.abs(epochs._data), threshold_percentile)
    epochs._data[:] = np.clip(epochs._data, -th, th)
    epochs.apply_baseline()
    return epochs
 
def segment_by_phoneme(raw):
    # preproc annotations
    meta = list()
    for annot in raw.annotations:
        d = eval(annot.pop("description")) # direct of the annotation
        for k, v in annot.items():
            assert k not in d.keys()
            d[k] = v
        meta.append(d)
    meta = pd.DataFrame(meta)
    meta["intercept"] = 1.0

    # compute voicing
    phonemes = meta.query('kind=="phoneme"') # find all phonemes
    assert len(phonemes)
    for ph, d in phonemes.groupby("phoneme"):
        ph = ph.split("_")[0]
        match = ph_info.query("phoneme==@ph")
        assert len(match) == 1
        meta.loc[d.index, "voiced"] = match.iloc[0].phonation == "v"

    # compute word frquency and merge w/ phoneme
    meta["is_word"] = False
    words = meta.query('kind=="word"').copy()
    assert len(words) > 10
    # assert np.all(meta.loc[words.index + 1, "kind"] == "phoneme")
    meta.loc[words.index + 1, "is_word"] = True
    wfreq = lambda x: zipf_frequency(x, "en")  # noqa
    # Create an index of word frequencies.
    meta.loc[words.index + 1, "wordfreq"] = words.word.apply(wfreq).values

    meta = meta.query('kind=="phoneme"')
    assert len(meta.wordfreq.unique()) > 2

    # segment
    events = np.c_[
        meta.onset * raw.info["sfreq"], np.ones((len(meta), 2))
    ].astype(int)

    epochs = mne.Epochs(
        raw,
        events,
        tmin=-0.200,
        tmax=0.6,
        decim=10,
        baseline=(-0.2, 0.0),
        metadata=meta,
        preload=True,
        event_repeated="drop",
    ) # Take the events and segment them

    # threshold
    th = np.percentile(np.abs(epochs._data), 95)
    # Treshold the data ?
    epochs._data[:] = np.clip(epochs._data, -th, th)
    epochs.apply_baseline()
    th = np.percentile(np.abs(epochs._data), 95)
    epochs._data[:] = np.clip(epochs._data, -th, th)
    epochs.apply_baseline()
    return epochs


def _word_epoch_words(word_meta):
    unique_words = list(word_meta["word"].unique())
    return S.filter_stop_words(unique_words)

def _get_raw_file(subject, session, task):
    print(".", end="")
    bids_path = mne_bids.BIDSPath(
        subject=subject,
        session=str(session),
        task=str(task),
        datatype="meg",
        root=MEG_MASC_ROOT
    )
    try:
        raw = mne_bids.read_raw_bids(bids_path)
    except FileNotFoundError:
        print("missing", subject, session, task)
        raise RuntimeError("missing %s, %s, %s" % (subject, session, task))
    raw = raw.pick_types(
        meg=True, misc=False, eeg=False, eog=False, ecg=False
    )
    # pick the frequency
    raw.load_data().filter(0.5, 30.0, n_jobs=1)
    return raw  

def _get_epochs(subject, segment=segment_by_phoneme):
    all_epochs = list()
    for session in range(2):
        for task in range(4):
            print(".", end="")
            bids_path = mne_bids.BIDSPath(
                subject=subject,
                session=str(session),
                task=str(task),
                datatype="meg",
                root=MEG_MASC_ROOT
            )
            try:
                raw = mne_bids.read_raw_bids(bids_path)
            except FileNotFoundError:
                print("missing", subject, session, task)
                continue
            raw = raw.pick_types(
                meg=True, misc=False, eeg=False, eog=False, ecg=False
            )
            # pick the frequency
            raw.load_data().filter(0.5, 30.0, n_jobs=1)

            epochs = segment(raw)
            epochs.metadata["half"] = np.round(
                np.linspace(0, 1.0, len(epochs))
            ).astype(int)
            epochs.metadata["task"] = task
            epochs.metadata["session"] = session

            all_epochs.append(epochs)
    if not len(all_epochs):
        return
    epochs = mne.concatenate_epochs(all_epochs)
    m = epochs.metadata
    label = (
        "t"
        + m.task.astype(str)
        + "_s"
        + m.session.astype(str)
        + "_h"
        + m.half.astype(str)
    )
    epochs.metadata["label"] = label
    return epochs
