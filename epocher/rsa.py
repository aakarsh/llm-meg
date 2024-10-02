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
from . import llm_bert as B


def _compare_rsa(similarity_matrix_0, similarity_matrix_1):
    # Assuming rsa_matrix_1 and rsa_matrix_2 are your similarity matrices
    correlation = np.corrcoef(similarity_matrix_0.flatten(), similarity_matrix_1.flatten())[0, 1]
    return correlation

def _compare_with_model(subject_id, task_id, session_id=0, model="GLOVE"):
    word_index, similarity_matrix = load_similarity_matrix(subject_id=subject_id, task_id=task_id)
    model_word_index, model_similarity_matrix = load_similarity_matrix(subject_id, task_id, model=model)
    # load model word index. make sure only intersection of 
    # these two are considered in model comparison. 
    # Ensure that same words are included ? 
    # Find common words between the human and model word indices
    common_words = list(set(word_index).intersection(set(model_word_index)))

    if not common_words:
        raise ValueError("No common words found between human and model word indices.")

    # Get the indices of the common words in both matrices
    word_index_positions = [word_index.index(word) for word in common_words]
    model_word_index_positions = [model_word_index.index(word) for word in common_words]

    # Subset the similarity matrices using the indices of common words
    human_similarity_submatrix = similarity_matrix[np.ix_(word_index_positions, word_index_positions)]
    model_similarity_submatrix = model_similarity_matrix[np.ix_(model_word_index_positions, model_word_index_positions)]

    # Compare the submatrices using RSA or other metrics
    return _compare_rsa(human_similarity_submatrix, model_similarity_submatrix)


def _compare_subjects(subject_id_1, subject_id_2, session_id=0, task_id=0, tmax=0.25):
    word_index, similarity_matrix_0 = _get_similarity_matrix(subject_id=subject_id_1, session_id=session_id, task_id=task_id, tmax=tmax)
    _, similarity_matrix_1 =  _get_similarity_matrix(subject_id=subject_id_2, 
            session_id=session_id, task_id=task_id, tmax=tmax, reference_word_idx = word_index)

    return word_index, _compare_rsa(similarity_matrix_0, similarity_matrix_1)

def _get_segmented_similarity_matrix(subject_id='01', session_id=0, task_id=0, 
                                        n_segments=10, n_components=15, tmax=0.25, 
                                        reference_word_idx = None, save_similarity_matrix=False, 
                                        debug=False):
    """Segmented siilarity matrices."""
     # Initialize dictionary to store ICA-transformed epochs
     word_index, word_metadata_df, word_epoch_map, ica_epochs = \
          D._get_ica_epochs(subject_id, session_id, task_id,n_components=n_components, tmax=tmax)

     # Overwrite word index with reference word index
     if reference_word_idx: 
          word_index = reference_word_idx
     
     pass

def average_word_occurances(word_index, word_epoch_map):
      # Extract ICA data for RSA
      target_word_vectors = []
      ica_avarage_map = {}
      for word in word_index:
        # print("word", word)
        epochs_ica = ica_epochs[word]
        # Average the ICA components over time
        ica_average_map[word] = epochs_ica.average().copy()

        # Flatten the data (optional: you can decide to not flatten depending on your approach)
        # print("vector",vector.shape, vector) 
      return ica_average_map

def _get_similarity_matrix(subject_id='01', session_id=0, task_id=0, n_components=15, tmax=0.25, 
        reference_word_idx = None, save_similarity_matrix=False, debug=False):

      # Initialize dictionary to store ICA-transformed epochs
      word_index, word_metadata_df, word_epoch_map, ica_epochs = \
          D._get_ica_epochs(subject_id, session_id, task_id,n_components=n_components, tmax=tmax)

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

      # convert to numpy array
      target_word_vectors = np.array(target_word_vectors)

      if debug:
          for i, vec in enumerate(target_word_vectors):
              print(f"word {i} vector (before normalization):", vec[:10])  # Check first 10 valuesA

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

# TODO: Slice the epochs.
# TODO: 1. Refactor the code above to resue the peoching
# TODO: 2. Crate windowed sections of the code above. 
# TODO: 3. Save the window id and interface to a similarity file.
# TODO: 4. For each window id and BERT Layer embedding compute the correlation score. 
# TODO: 5. Plot the Correlation Coefficient between each BERT Layer and the time window which it explains most.
# Assumning each word is devided into 100 onsite times, then we will have 100 rsa which compare word in that window

def compute_similarity_matrics(subject_id, task_id, model="GLOVE", save_similarity_matrix=True):
    word_index = load_word_index(subject_id, task_id)
    similarity_matrix = None
    if model == "GLOVE":
          similarity_matrix = G.create_rsa_matrix(word_index)
    elif model == "BERT":
          # some words not found
          word_index, similarity_matrix = B.create_rsa_matrix(word_index, task_id)
    else:
        raise RuntimeError(f'Unkown model: {model}')

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

def load_word_index(subject_id, task_id, model=None, output_dir = OUTPUT_DIR):
    word_index_file = f'{output_dir}/subject_{subject_id}_task_{task_id}_word_index.json'
    if model: 
        word_index_file = f'{output_dir}/model_{model}_subject_{subject_id}_task_{task_id}_word_index.json'
    word_index = None
    with open(word_index_file, 'r') as infile:
         word_index = json.load(infile)
    return word_index

def load_model_similarity_matrix(subject_id, task_id, model):
    similarity_matrix_file = f'{OUTPUT_DIR}/model_{model}_subject_{subject_id}_task_{task_id}_similarity_matrix.npy'
    similarity_matrix = np.load(similarity_matrix_file)

    return similarity_matrix

def load_similarity_matrix(subject_id, task_id, model=None):
    similarity_matrix_file = f'{OUTPUT_DIR}/subject_{subject_id}_task_{task_id}_similarity_matrix.npy'
    if model: 
        similarity_matrix_file = f'{OUTPUT_DIR}/model_{model}_subject_{subject_id}_task_{task_id}_similarity_matrix.npy'
    word_index = load_word_index(subject_id, task_id, model=model)
    similarity_matrix = np.load(similarity_matrix_file)

    return word_index, similarity_matrix
 
