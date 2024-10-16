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

import mne
from mne.preprocessing import ICA
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity


from .env import *
from . import stories as S 

# Create a BIDSPath object
DATASET_ROOT="/content/drive/MyDrive/TUE-SUMMER-2024/ulm-meg/"

ph_info = pd.read_csv(MEG_MASC_ROOT+"/phoneme_info.csv")

def load_bids_path(root=DATASET_ROOT, subject="01", datatype="meg", session="0",  task="1"):
    bids_path = BIDSPath(root=DATASET_ROOT, subject=subject, session=session, task=task,datatype=datatype)
    return bids_path

def _load_epoch_map(subject_id='01', session_id=0, task_id=0, n_components=15, 
                                tmax=0.25, reference_word_idx=None, 
                                word_pos=['VB'], use_ica=True):
      # Initialize dictionary to store ICA-transformed epochs
      word_index, word_metadata_df, word_epoch_map, ica_epochs  = None, None, None, None
      current_word_epoch_map = {}
      if use_ica:
          word_index, word_metadata_df, word_epoch_map, ica_epochs = \
              _get_ica_epochs(subject_id, session_id, task_id,
                                  n_components=n_components, tmax=tmax, word_pos=word_pos)
          current_word_epoch_map = ica_epochs
      else:
          word_index, word_metadata_df, word_epoch_map = \
              _get_epoch_word_map(subject_id, session_id, task_id, 
                      tmax=tmax, word_pos=word_pos)
          current_word_epoch_map = word_epoch_map
          for word, epochs in current_word_epoch_map.items(): 
              baseline_period = (-0.2, 0)
              # Apply baseline correction to the ICA-transformed data
              epochs.apply_baseline(baseline=baseline_period)
      return word_index, current_word_epoch_map


def _get_target_word_vectors_per_electrode(subject_id='01', session_id=0, task_id=0, n_components=15, 
                                tmax=0.25, reference_word_idx=None, 
                                word_pos=['VB'], use_ica=False):

      # Initialize dictionary to store ICA-transformed epochs
      word_index, current_word_epoch_map = _load_epoch_map(subject_id, session_id, task_id, 
              n_components=n_components, tmax=tmax, word_pos=word_pos, use_ica=use_ica) 

      # Overwrite word index with reference word index
      if reference_word_idx: 
          word_index = reference_word_idx

      # Extract ICA data for RSA
      target_word_vectors = []

      for word in word_index:
        # print("word", word)
        word_epochs = current_word_epoch_map[word]
        # Average the ICA components over time
        epoch_average = word_epochs.average().get_data()  # Shape: (n_channels, n_times)

        # TODO Don't flatten the data , but rather keep the elecrode dimensions
        # Flatten the data (optional: you can decide to not flatten depending on your approach)
        vector = epoch_average.flatten()
        target_word_vectors.append(vector)
        # print("vector",vector.shape, vector) 

      # convert to numpy array
      target_word_vectors = np.array(target_word_vectors)

      return word_index, target_word_vectors


def _get_target_word_vectors(subject_id='01', session_id=0, task_id=0, n_components=15, 
                                tmax=0.25, reference_word_idx=None, 
                                word_pos=['VB'], use_ica=False):

      # Initialize dictionary to store ICA-transformed epochs
      word_index, current_word_epoch_map = _load_epoch_map(subject_id, session_id, task_id,
                                                              n_components=n_components, tmax=tmax, 
                                                              word_pos=word_pos, use_ica=use_ica) 
      # Overwrite word index with reference word index
      if reference_word_idx: 
          word_index = reference_word_idx

      # Extract ICA data for RSA
      target_word_vectors = []

      for word in word_index:
        # print("word", word)
        word_epochs = current_word_epoch_map[word]
        # Average the ICA components over time
        epoch_average = word_epochs.average().get_data()  # Shape: (n_channels, n_times)

        # Flatten the data (optional: you can decide to not flatten depending on your approach)
        vector = epoch_average.flatten()
        target_word_vectors.append(vector)
        # print("vector",vector.shape, vector) 

      # convert to numpy array
      target_word_vectors = np.array(target_word_vectors)
      return word_index, target_word_vectors


def _get_ica_epochs(subject_id='01', session_id=0, task_id=0, 
        n_components=15, tmax=0.25, word_pos=['VB']):
      """
      ICA: Aggregate for same task accross sessions. 
      """
      if type(session_id) == list:
          # Given  a task list.
          session_id = session_id[0]

      word_index, word_metadata_df, word_epoch_map = \
          _get_epoch_word_map(subject_id, session_id, task_id, 
                  tmax=tmax, word_pos=word_pos)

      # Initialize dictionary to store ICA-transformed epochs
      ica_epochs = {}

      # Initialize ICA model
      ica = ICA(n_components=n_components, random_state=42)

      # Loop through each target word and apply ICA
      for word in word_index:
        epochs = word_epoch_map[word]
          # Get the epoch data for the target word
        if len(epochs) == 0:
          continue
        # Fit ICA to the epochs data
        ica.fit(epochs)

        # Apply ICA to the epochs to get independent components
        epochs_ica = ica.apply(epochs.copy())

        # Baseline-correct the ICA-transformed epochs
        # Define the baseline period. For example, (-0.2, 0) takes the 
        # time period between -200 ms and 0 ms.
        baseline_period = (-0.2, 0)
        # Apply baseline correction to the ICA-transformed data
        epochs_ica.apply_baseline(baseline=baseline_period)

        # Store the ICA-transformed epochs
        ica_epochs[word] = epochs_ica
        
      return word_index, word_metadata_df, word_epoch_map, ica_epochs


def _get_epoch_word_map(subject_id, session_id, task_id, 
                            tmax=0.25, word_pos=["VB"]):
    """ 
    We use a 250 ms window.
    """
    raw_file = _get_raw_file(subject_id, session_id, task_id)

    word_epochs = segment_by_word(raw_file, tmax=tmax)
    words_meta = word_epochs.metadata

    # Filter out the stop words.
    words_found = _word_epoch_words(words_meta, word_pos=word_pos)

    words_found_metadata_df =  words_meta[words_meta["word"].isin(words_found)]
    words_sorted_metadata_df = words_found_metadata_df.sort_values(by="word")
    words_sorted_index = words_sorted_metadata_df.index

    target_word_epochs = { word: word_epochs[words_meta[words_meta["word"] == word].index] for word in words_found 
            if len(word_epochs[words_meta[words_meta["word"] == word].index]) >0 }
   
    word_index = list(sorted(target_word_epochs.keys()))
    
    return word_index, words_sorted_metadata_df, target_word_epochs 

def _segment_word_epoch_map(num_segments, word_index,  target_word_epochs):
    """
    We want to segment the word epochs by chunks of certain-segment duration. Thus 
    instead of a single word_epoch we will have multiple epochs, each one will 
    then be individually used to generate a RSA_map.
    """
    segmented_epochs = {}
    for word in word_index:
        # Get the epochs for the word
        word_epoch = target_word_epochs[word]
        
        # Get the time points in the epoch (this assumes all epochs have the same time range)
        tmin, tmax = word_epoch.tmin, word_epoch.tmax
        total_time = tmax - tmin
        
        # Divide the total time by the number of segments
        segment_duration = total_time / num_segments
        
        # Define the segment boundaries (sliding windows)
        for i in range(num_segments):
            segment_tmin = tmin + i * segment_duration
            segment_tmax = segment_tmin + segment_duration
            
            # Create new segmented epoch
            segmented_epoch = word_epoch.copy().crop(tmin=segment_tmin, tmax=segment_tmax)
            
            # Add to the dictionary
            if word not in segmented_epochs:
                segmented_epochs[word] = []
            segmented_epochs[word].append(segmented_epoch)
    
    return segmented_epochs

def parse_event_description(event_json_str):
    event_json_str = event_json_str.replace('\'', '"')
    return json.loads(event_json_str)

def load_subject_information():
    # Read information about subjects
    subjects = pd.read_csv(MEG_MASC_ROOT +  "/participants.tsv", sep="\t")
    return subjects

def load_subject_ids():
    # find subject id
    subjects = load_subject_information()
    subjects = subjects.participant_id.apply(lambda x: x.split("-")[1]).values
    subjects = subjects[0:11]
    return subjects 

def load_task_ids():
    return [0, 1, 2, 3]

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


def segment_by_word(raw, tmax=0.25):
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
                                baseline=None, #(-0.2,0.0), 
                                metadata=meta, 
                                preload=True, 
                                event_repeated="drop")
    #word_epochs = _threshold_baseline_epochs(word_epochs)
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


def _word_epoch_words(word_meta, word_pos=['VB']):
    unique_words = list(word_meta["word"].unique())
    lower_case_unique_words = list(map(lambda s : s.lower(), unique_words))
    selected_words = S.select_words_by_part_of_speech(lower_case_unique_words, word_pos=word_pos)
    return selected_words

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

def sliding_window_rsa_per_electrode(subject_id='01', session_id=0, task_id=0, 
    window_size=0.1, step_size=0.05, word_pos=['VB'], use_ica=False):
    """
    Perform per-electrode sliding window RSA using existing epoch map.
    
    Args:
    - subject_id: Subject identifier.
    - session_id: Session identifier.
    - task_id: Task identifier.
    - window_size: Size of the sliding window in seconds.
    - step_size: Step size for the sliding window in seconds.
    - word_pos: List of parts of speech tags to consider.
    - use_ica: Boolean indicating if ICA-transformed data should be used.
    
    Returns:
    - rsa_values: A dictionary with keys as time points and values as RSA values for all electrodes.
    """
    
    # Load word epochs for each word using your existing method
    word_index, word_epoch_map = _load_epoch_map(subject_id, session_id, task_id, 
            use_ica=use_ica, word_pos=word_pos)

    # Initialize the sliding window RSA
    sfreq = word_epoch_map[word_index[0]].info['sfreq']  # Sampling frequency from the epochs
    window_samples = int(window_size * sfreq)
    step_samples = int(step_size * sfreq)
    
    rsa_values = []  # Will store RSA values for each time window
    time_points = []

    # Loop through all words
    for word in word_index:
        epochs = word_epoch_map[word]  # Shape: (n_epochs, n_channels, n_times)

        if len(epochs) == 0:
            continue

        n_channels = len(epochs.ch_names)
        n_times = len(epochs.times)

        # Perform sliding window over the time dimension for each electrode
        for start in range(0, n_times - window_samples + 1, step_samples):
            end = start + window_samples
            
            # Average over trials for the current time window.
            epoch_window = epochs.get_data()[:, :, start:end]  # Shape: (n_epochs, n_channels, window_samples)
            avg_window = np.mean(epoch_window, axis=0)  # Shape: (n_channels, window_samples)
            
            rsa_per_electrode = []
            
            # Compute RSA for each electrode
            for ch in range(n_channels):
                electrode_vector = avg_window[ch, :].flatten()
                norm = np.linalg.norm(electrode_vector) + 1e-10  # Avoid division by zero
                normalized_vector = electrode_vector / norm
                # Ok the self similarity part is nutso. 
                rsa_value = np.corrcoef(normalized_vector, normalized_vector)[0, 1]  # Self-similarity here for simplicity
                rsa_per_electrode.append(rsa_value)
            
            rsa_values.append(rsa_per_electrode)
            time_points.append(epochs.times[start])

    return np.array(rsa_values), time_points
