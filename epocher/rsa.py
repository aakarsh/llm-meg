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
from . import dataset as D


def _get_ica_epochs(subject_id='01', session_id=0, task_id=0, n_components=40):
      word_index, word_metadata_df, word_epoch_map = D._get_epoch_word_map(subject_id, session_id, task_id)
      # Initialize dictionary to store ICA-transformed epochs
      ica_epochs = {}

      # Loop through each target word and apply ICA
      for word in word_index:
        print("word", word)
        epochs = word_epoch_map[word]
          # Get the epoch data for the target word
        if len(epochs) == 0:
          continue
        # Initialize ICA model
        ica = ICA(n_components=n_components, random_state=42)

        # Fit ICA to the epochs data
        ica.fit(epochs)

        # Apply ICA to the epochs to get independent components
        epochs_ica = ica.apply(epochs.copy())

        # Store the ICA-transformed epochs
        ica_epochs[word] = epochs_ica
        print("epoch_ica", word, epochs_ica.average().get_data())

      return word_index, word_metadata_df, word_epoch_map, ica_epochs
def _get_similarity_matrix(subject_id='01', session_id=0, task_id=0):

      word_index, word_metadata_df, word_epoch_map = D._get_epoch_word_map(subject_id, session_id, task_id)
      # Initialize dictionary to store ICA-transformed epochs
      ica_epochs = {}

      # Loop through each target word and apply ICA
      for word in word_index:
        print("word", word)
        epochs = word_epoch_map[word]
          # Get the epoch data for the target word
        if len(epochs) == 0:
          continue
        # Initialize ICA model
        ica = ICA(n_components=20, random_state=42)

        # Fit ICA to the epochs data
        ica.fit(epochs)

        # Apply ICA to the epochs to get independent components
        epochs_ica = ica.apply(epochs.copy())

        # Store the ICA-transformed epochs
        ica_epochs[word] = epochs_ica
        print("epoch_ica", word, epochs_ica.average().get_data())

      # Extract ICA data for RSA
      target_word_vectors = []

      for word in word_index:
        print("word", word)
        epoch_ica = ica_epochs[word]
        # Average the ICA components over time
        avg_ica = epochs_ica.average().get_data()  # Shape: (n_channels, n_times)

        # Flatten the data (optional: you can decide to not flatten depending on your approach)
        vector = avg_ica.flatten()
        target_word_vectors.append(vector)
        print("vector",vector.shape, vector) 
      # Convert to numpy array
      target_word_vectors = np.array(target_word_vectors)

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
      similarity_matrix = cosine_similarity(target_word_vectors)

      return word_index, similarity_matrix
