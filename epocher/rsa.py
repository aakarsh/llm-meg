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
import logging

import mne
from mne.preprocessing import ICA
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity


from .env import *
from . import stories as S 
from . import dataset as D
from . import llm_glove as G


def _get_ica_epochs(subject_id='01', session_id=0, task_id=0, n_components=15, tmax=0.25):
      """
      ICA: Aggregate for same task accross sessions. 
      """
      if type(session_id) == list:
          # Given  a task list.
          session_id = session_id[0]

      word_index, word_metadata_df, word_epoch_map = \
          D._get_epoch_word_map(subject_id, session_id, task_id, tmax=tmax)

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


def _compare_rsa(similarity_matrix_0, similarity_matrix_1):
    # Assuming rsa_matrix_1 and rsa_matrix_2 are your similarity matrices
    correlation = np.corrcoef(similarity_matrix_0.flatten(), similarity_matrix_1.flatten())[0, 1]
    return correlation

def _compare_subjects(subject_id_1, subject_id_2, session_id=0, task_id=0, tmax=0.25):
    word_index, similarity_matrix_0 = _get_similarity_matrix(subject_id=subject_id_1, session_id=session_id, task_id=task_id, tmax=tmax)
    _, similarity_matrix_1 =  _get_similarity_matrix(subject_id=subject_id_2, 
            session_id=session_id, task_id=task_id, tmax=tmax, reference_word_idx = word_index)

    return word_index, _compare_rsa(similarity_matrix_0, similarity_matrix_1)


def _get_similarity_matrix(subject_id='01', session_id=0, task_id=0, n_components=15, tmax=0.25, 
        reference_word_idx = None, save_similarity_matrix=False, debug=False):

      # Initialize dictionary to store ICA-transformed epochs
      word_index, word_metadata_df, word_epoch_map, ica_epochs = \
          _get_ica_epochs(subject_id, session_id, task_id,n_components=n_components, tmax=tmax)

      # Overwrite word index with reference word index
      if reference_word_idx: 
          word_index = reference_word_idx

      # Extract ICA data for RSA
      target_word_vectors = []

      for word in word_index:
        # print("word", word)
        epochs_ica = ica_epochs[word]
        # Average the ICA components over time
        avg_ica = epochs_ica.average().get_data()  # Shape: (n_channels, n_times)

        # Flatten the data (optional: you can decide to not flatten depending on your approach)
        vector = avg_ica.flatten()
        target_word_vectors.append(vector)
        # print("vector",vector.shape, vector) 
      # Convert to numpy array
      target_word_vectors = np.array(target_word_vectors)

      if debug:
          for i, vec in enumerate(target_word_vectors):
              print(f"Word {i} vector (before normalization):", vec[:10])  # Check first 10 valuesA

          for word, epochs_ica in ica_epochs.items():
              print(f"ICA data for {word}: {epochs_ica.get_data().shape}")

      # Normalize each word vector across its flattened dimensions (n_channels, n_padded_times, n_trials)
      normalized_vectors = []
      for vec in target_word_vectors:
        vec_flat = vec.flatten()  # Flatten the vector
        norm = np.linalg.norm(vec_flat)  # Compute the L2 norm
        normalized_vec = vec_flat / norm  # Normalize the vector
        normalized_vectors.append(normalized_vec)

      # Convert normalized vectors to numpy array
      normalized_vectors = np.array(normalized_vectors)

      # Compute cosine similarity matrix
      similarity_matrix = cosine_similarity(normalized_vectors)


      if save_similarity_matrix: 
          # Serialize the word index as JSON
          word_index_file = f'{OUTPUT_DIR}/subject_{subject_id}_task_{task_id}_word_index.json'
          with open(word_index_file, 'w') as f:
              json.dump(word_index, f)
          #Serialize the similarity matrix as an `.npy` file
          similarity_matrix_file = f'{OUTPUT_DIR}/subject_{subject_id}_task_{task_id}_similarity_matrix.npy'
          np.save(similarity_matrix_file, similarity_matrix)
      return word_index, similarity_matrix


def compute_similarity_matrics(subject_id, task_id, model="GLOVE", save_similarity_matrix=True):
    word_index = load_word_index(subject_id, task_id)
    similarity_matrix = None
    if model == "GLOVE":
          similarity_matrix = G.create_rsa_matrix(word_index)
          if save_similarity_matrix: 
              # Serialize the word index as JSON
              word_index_file = f'{OUTPUT_DIR}/model_{model}_subject_{subject_id}_task_{task_id}_word_index.json'
              with open(word_index_file, 'w') as f:
                  json.dump(word_index, f)
              # serialize the similarity matrix as an `.npy` file
              similarity_matrix_file = f'{OUTPUT_DIR}/model_{model}_subject_{subject_id}_task_{task_id}_similarity_matrix.npy'
              np.save(similarity_matrix_file, similarity_matrix)
              print(f'Created {similarity_matrix_file}')
    return similarity_matrix  

def load_word_index(subject_id, task_id, output_dir = OUTPUT_DIR):
    word_index_file = f'{output_dir}/subject_{subject_id}_task_{task_id}_word_index.json'
    word_index = None
    with open(word_index_file, 'r') as infile:
         word_index = json.load(infile)
    return word_index


def load_similarity_matrix(subject_id, task_id):
    similarity_matrix_file = f'{OUTPUT_DIR}/subject_{subject_id}_task_{task_id}_similarity_matrix.npy'
    word_index = load_word_index(subject_id, task_id)
    similarity_matrix = np.load(similarity_matrix_file)

    return word_index, similarity_matrix
 
