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

# TODO: Slice the epochs:
# TODO: 1.  Refactor the code above to resue the epoching
# TODO: 2.  Create windowed sections of the code above. 
# TODO: 3.  Save the window id and interface to a similarity file.
# TODO: 4.  For each window id and BERT Layer embedding compute the correlation score. 
# TODO: 5.  Plot the Correlation Coefficient between each BERT Layer and the time window which it explains most.
# TODO: 6.  Perform per-electrode correlations 
# TODO: 7.  Look at doing noun comparisons.
# TODO: 8.  Look at doing functional part of speech comparisons
# TODO: 9.  Look at doing boot-strapping and Cross-validation
# TODO: 10. Look at encoding models.
# TODO: 10. Look at sentence concept. 

# Assumning each word is devided into 100 onsite times, then we will have 100 rsa which compare word in that window


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

def _compare_segemnts_with_model_layers(subject_id, task_id, session_id=0, model="BERT"):
    retval_similarity_matrix = np.zeros((20, 12))

    for segment_idx in range(1, 20):
        for layer_idx in range( 12, 1, -1):
            word_index, similarity_matrix = load_similarity_matrix(subject_id=subject_id, task_id=task_id, segmented=True)
            model_word_index, model_similarity_matrix = load_similarity_matrix(subject_id, task_id, model=model, layer_id=layer_idx)
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
            similarity_matrix=similarity_matrix[segment_idx]
            human_similarity_submatrix = similarity_matrix[np.ix_(word_index_positions, word_index_positions)]
            model_similarity_submatrix = model_similarity_matrix[np.ix_(model_word_index_positions, model_word_index_positions)]

            # Compare the submatrices using RSA or other metrics
            retval_similarity_matrix[segment_idx-1, layer_idx-1] = _compare_rsa(human_similarity_submatrix, model_similarity_submatrix)
    print(f"Final-Similairty subject-id{subject_id} {task_id} {session_id}", similarity_matrix)

    def plot_heatmap(rsa_matrix, title='RSA Comparison Heatmap'):
        """Plot a heatmap of the RSA comparison matrix."""
        plt.figure(figsize=(64, 32))
        sns.heatmap(rsa_matrix.T, annot=True, fmt=".2f", cmap='coolwarm', cbar_kws={"shrink": .8})
        plt.title(title)
        plt.ylabel('BERT Layers')
        plt.xlabel('MEG Segments')
        plt.savefig(f'./images/segment-similarity_matrix-subject_id-{subject_id}-task_id-{task_id}.png')
        plt.close()

    plot_heatmap(retval_similarity_matrix)
    return retval_similarity_matrix 


def _compare_subjects(subject_id_1, subject_id_2, session_id=0, task_id=0, tmax=0.25):
    word_index, similarity_matrix_0 = _get_similarity_matrix(subject_id=subject_id_1, session_id=session_id, 
            task_id=task_id, tmax=tmax)
    _, similarity_matrix_1 =  _get_similarity_matrix(subject_id=subject_id_2, 
            session_id=session_id, task_id=task_id, tmax=tmax, reference_word_idx = word_index)

    return word_index, _compare_rsa(similarity_matrix_0, similarity_matrix_1)



def _get_sliding_window_rsa(subject_id='01', session_id=0, task_id=0, 
        n_segments=20, n_components=15, tmax=0.25):

    # Load your MEG raw data here 
    raw_data = load_raw_data(subject_id, task_id)
    windowed_similarity = []
    
    # Assuming you already have the word events processed
    for start in range(0, raw_data.times[-1] - window_size + 1, step_size):
        end = start + window_size
        # Extract MEG data for this window
        data_window = raw_data.copy().crop(tmin=start, tmax=end)
        # Compute similarity matrix for this window
        similarity_matrix = compute_similarity_matrix(data_window)
        
        # Calculate RSA value with reference
        rsa_value = _compare_rsa(similarity_matrix, ref_rsa)
        windowed_similarity.append(rsa_value)

    # Optionally, plot the results for continuous RSA values over time
    # plt.plot(range(0, len(windowed_similarity) * step_size, step_size), windowed_similarity)
    # plt.xlabel("Time (ms)")
    # plt.ylabel("RSA Similarity")
    # plt.title("Sliding Window RSA Similarity Over Time")
    # plt.show()
    return windowed_similarity

def _get_segmented_similarity_matrix(subject_id='01', session_id=0, task_id=0, 
                                        n_segments=20, n_components=15, tmax=0.25, 
                                        reference_word_idx = None, save_similarity_matrix=False, 
                                        debug=False):
    """
        Segmented siilarity matrices.
    """
    # Initialize dictionary to store ICA-transformed epochs
    # TODO - We can't use ICA, should use oly for artifact removal. 
    # Need to do something else. 
    word_index, word_metadata_df, word_epoch_map, ica_epochs = \
         D._get_ica_epochs(subject_id, session_id, task_id,
                                  n_components=n_components, tmax=tmax)
    # Overwrite word index with reference word index
    if reference_word_idx: 
         word_index = reference_word_idx

    ica_average_map = average_word_occurances(word_index, word_epoch_map)
    segmented_epochs = D._segment_word_epoch_map(n_segments, word_index, ica_average_map)
    similarity_matrices = []

    for segment_idx in range(n_segments):
       target_word_vectors = []
       for word in word_index:
           epochs_ica = segmented_epochs[word][segment_idx]
           avg_ica = epochs_ica.get_data()
           # Flatten the data (optional: you can decide to not flatten depending on your approach)
           vector = avg_ica.flatten()
           target_word_vectors.append(vector)

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
       similarity_matrices.append(cosine_similarity(normalized_vectors))

    segmented_similarity_matrices = np.array(similarity_matrices)
    if save_similarity_matrix: 
      # Serialize the word index as JSON
      word_index_file = f'{OUTPUT_DIR}/segmented-subject_{subject_id}_task_{task_id}_word_index.json'
      with open(word_index_file, 'w') as f:
          json.dump(word_index, f)
      #Serialize the similarity matrix as an `.npy` file
      similarity_matrix_file = f'{OUTPUT_DIR}/segmented-subject_{subject_id}_task_{task_id}_similarity_matrix.npy'
      np.save(similarity_matrix_file, segmented_similarity_matrices)

    return word_index, segmented_similarity_matrices


def average_word_occurances(word_index, ica_epochs):
      # Extract ICA data for RSA
      target_word_vectors = []
      ica_average_map = {}
      for word in word_index:
        epochs_ica = ica_epochs[word]
        # Average the ICA components over time
        ica_average_map[word] = epochs_ica.average().copy()
      return ica_average_map

def _get_per_per_electrode(subject_id='01', session_id=0, task_id=0, n_components=15, tmax=0.25, 
        reference_word_idx=None, save_similarity_matrix=False, word_pos=['VB'], debug=False):
        """
        Show the working on a per-electrode basis.
        """
    pass


def _get_similarity_matrix(subject_id='01', session_id=0, task_id=0, n_components=15, tmax=0.25, 
        reference_word_idx=None, save_similarity_matrix=False, word_pos=['VB'], debug=False):

      word_index, target_word_vectors = \
          D._get_target_word_vectors(subject_id, session_id, task_id,
                                          reference_word_idx = reference_word_idx,
                                          n_components=n_components, 
                                          tmax=tmax, 
                                          word_pos=word_pos, 
                                          use_ica=False)

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
          pos_tag =  "_".join(word_pos)
          word_index_file = f'{OUTPUT_DIR}/subject_{subject_id}_task_{task_id}_pos_{pos_tag}_word_index.json'
          with open(word_index_file, 'w') as f:
              json.dump(word_index, f)
          #Serialize the similarity matrix as an `.npy` file
          similarity_matrix_file = f'{OUTPUT_DIR}/subject_{subject_id}_task_{task_id}_pos_{pos_tag}_similarity_matrix.npy'
          np.save(similarity_matrix_file, similarity_matrix)

      return word_index, similarity_matrix


def compute_similarity_matrics(subject_id, task_id, model="GLOVE", hidden_layer=-1, save_similarity_matrix=True):
    word_index = load_word_index(subject_id, task_id)
    similarity_matrix = None
    if model == "GLOVE":
          similarity_matrix = G.create_rsa_matrix(word_index)
    elif model == "BERT":
          # some words not found
          word_index, similarity_matrix = B.create_rsa_matrix(word_index, task_id, hidden_layer=hidden_layer)
    else:
        raise RuntimeError(f'Unkown model: {model}')

    layer_tag = ""
    if model == "BERT" and hidden_layer!=-1:
        layer_tag = f"_layer_{hidden_layer}"
    if save_similarity_matrix: 
      # Serialize the word index as JSON
      word_index_file = f'{OUTPUT_DIR}/model_{model}_{layer_tag}_subject_{subject_id}_task_{task_id}_word_index.json'
      with open(word_index_file, 'w') as f:
          json.dump(word_index, f)
      # serialize the similarity matrix as an `.npy` file
      similarity_matrix_file = f'{OUTPUT_DIR}/model_{model}_{layer_tag}_subject_{subject_id}_task_{task_id}_similarity_matrix.npy'
      np.save(similarity_matrix_file, similarity_matrix)
      print(f'Created {similarity_matrix_file}')
    return similarity_matrix  

def load_word_index(subject_id, task_id, model=None, output_dir = OUTPUT_DIR, segmented=False, layer_id=False):
    word_index_file = f'{output_dir}/subject_{subject_id}_task_{task_id}_word_index.json'
    if model: 
        word_index_file = f'{output_dir}/model_{model}_subject_{subject_id}_task_{task_id}_word_index.json'
        if layer_id: 
            word_index_file = f'{output_dir}/model_{model}__layer_-{layer_id}_subject_{subject_id}_task_{task_id}_word_index.json'

    if segmented:
        word_index_file = f'{output_dir}/segmented-subject_{subject_id}_task_{task_id}_word_index.json'

    word_index = None
    with open(word_index_file, 'r') as infile:
         word_index = json.load(infile)
    return word_index

def load_model_similarity_matrix(subject_id, task_id, model):
    similarity_matrix_file = f'{OUTPUT_DIR}/model_{model}_subject_{subject_id}_task_{task_id}_similarity_matrix.npy'
    similarity_matrix = np.load(similarity_matrix_file)

    return similarity_matrix

def load_similarity_matrix(subject_id, task_id, model=None, segmented=False, layer_id=None):
    similarity_matrix_file = f'{OUTPUT_DIR}/subject_{subject_id}_task_{task_id}_similarity_matrix.npy'
    if model:
        if layer_id:
            similarity_matrix_file = f'{OUTPUT_DIR}/model_{model}__layer_-{layer_id}_subject_{subject_id}_task_{task_id}_similarity_matrix.npy'
        else:
            similarity_matrix_file = f'{OUTPUT_DIR}/model_{model}_subject_{subject_id}_task_{task_id}_similarity_matrix.npy'
    if segmented:
            similarity_matrix_file = f'{OUTPUT_DIR}/segmented-subject_{subject_id}_task_{task_id}_similarity_matrix.npy'
            
    word_index = load_word_index(subject_id, task_id, model=model, segmented=segmented, layer_id=layer_id)
    similarity_matrix = np.load(similarity_matrix_file)

    return word_index, similarity_matrix
 
