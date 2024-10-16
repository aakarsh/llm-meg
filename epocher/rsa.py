from mne_bids import BIDSPath 
from mne.preprocessing import ICA
from pathlib import Path
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, scale

from tqdm import trange

from wordfreq import zipf_frequency

import json
import logging
import matplotlib
import matplotlib.pyplot as plt
import mne
import mne_bids
import numpy as np
import os 
import pandas as pd
import seaborn as sns
import scipy.cluster.hierarchy as sch
from sklearn.manifold import SpectralEmbedding


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

# Helper function to check file existence
def _file_exists(filepath):
    """Check if a file exists at the given filepath."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

def _save_to_file(data, filename):
    """Save data to file in either JSON or NumPy format based on extension."""
    if filename.endswith('.json'):
        with open(filename, 'w') as f:
            json.dump(data, f)
    elif filename.endswith('.npy'):
        np.save(filename, data)
    else:
        raise ValueError("Unsupported file format. Use .json or .npy")


def _load_from_file(filename):
    """Load data from a file in either JSON or NumPy format based on extension."""
    _file_exists(filename)
    if filename.endswith('.json'):
        with open(filename, 'r') as f:
            return json.load(f)
    elif filename.endswith('.npy'):
        return np.load(filename)
    else:
        raise ValueError("Unsupported file format. Use .json or .npy")


def _compare_rsa(similarity_matrix_0, similarity_matrix_1):
    # Assuming rsa_matrix_1 and rsa_matrix_2 are your similarity matrices
    correlation = np.corrcoef(similarity_matrix_0.flatten(), similarity_matrix_1.flatten())[0, 1]
    return correlation


def _compare_with_model(subject_id, task_id, session_id=0, model="GLOVE", word_pos=['VB']):
    word_index, similarity_matrix = load_similarity_matrix(subject_id=subject_id, task_id=task_id, word_pos=word_pos)
    model_word_index, model_similarity_matrix = load_similarity_matrix(subject_id, task_id, model=model, word_pos=word_pos)
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

def _compare_segments_with_model_layers(subject_id, task_id, session_id=0, model="BERT"):
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
            similarity_matrix = similarity_matrix[segment_idx]
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
                                        reference_word_idx=None, save_similarity_matrix=False, 
                                        debug=False, word_pos=['VB']):
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
            norm = np.linalg.norm(vec_flat) + 1e-10  # Compute the L2 norm
            normalized_vec = vec_flat / norm  # Normalize the vector
            normalized_vectors.append(normalized_vec)

       # Convert normalized vectors to numpy array
       normalized_vectors = np.array(normalized_vectors)

       # Compute cosine similarity matrix
       similarity_matrices.append(cosine_similarity(normalized_vectors))

    segmented_similarity_matrices = np.array(similarity_matrices)
    if save_similarity_matrix: 
        save_similarity_data(word_index, segmented_similarity_matrices, 
                subject_id, task_id, segmented=True, word_pos=word_pos)
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

def _get_per_electrode_similarity_matrix(subject_id='01', session_id=0, task_id=0, n_components=15, tmax=0.25, 
        reference_word_idx=None, save_similarity_matrix=False, word_pos=['VB'], debug=False):
    word_index, target_word_vectors = \
          D._get_target_word_vectors(subject_id, session_id, task_id,
                                          reference_word_idx=reference_word_idx,
                                          n_components=n_components, 
                                          tmax=tmax, 
                                          word_pos=word_pos, 
                                          use_ica=False)
    print(f"per-electrode-rsa: {target_word_vectors.shape}, word_index: {len(word_index)}")


def _get_similarity_matrix(subject_id='01', session_id=0, task_id=0, n_components=15, tmax=0.25, 
        reference_word_idx=None, save_similarity_matrix=False, word_pos=['VB'], debug=False):

      word_index, target_word_vectors = \
          D._get_target_word_vectors(subject_id, session_id, task_id,
                                          reference_word_idx=reference_word_idx,
                                          n_components=n_components, 
                                          tmax=tmax, 
                                          word_pos=word_pos, 
                                          use_ica=False)

      # Normalize each word vector across its flattened dimensions (n_channels, n_padded_times, n_trials)
      normalized_vectors = []
      for vec in target_word_vectors:
        vec_flat = vec.flatten()  # Flatten the vector
        norm = np.linalg.norm(vec_flat) + 1e-10  # Compute the L2 norm
        normalized_vec = vec_flat / norm  # Normalize the vector
        normalized_vectors.append(normalized_vec)

      # Convert normalized vectors to numpy array
      normalized_vectors = np.array(normalized_vectors)

      # Compute cosine similarity matrix
      similarity_matrix = cosine_similarity(normalized_vectors)

      if save_similarity_matrix: 
        save_similarity_data(word_index, similarity_matrix, subject_id, task_id,  word_pos=word_pos)

      return word_index, similarity_matrix


def compute_similarity_matrics(subject_id, task_id, model="GLOVE", 
        hidden_layer=None, word_pos=['VB'], save_similarity_matrix=True):
    """
    Compute the simialirty matrix for given task given a model.
    """
    # FIX: We are saving wordlist per subject, but do we need to ?
    word_index = load_word_index(subject_id, task_id, word_pos=word_pos) 
    similarity_matrix = None

    if model == "GLOVE": # FIX: why would i need subject_id
      similarity_matrix = G.create_rsa_matrix(word_index)
  elif model == "BERT": # FIX: why would i need subject_id
      # Some words not found
      word_index, similarity_matrix = B.create_rsa_matrix(word_index, task_id, 
                  hidden_layer=hidden_layer)
    else:
        raise RuntimeError(f'Unkown model: {model}')
    
    if save_similarity_matrix: # why would i need subject_id 
      save_similarity_data(word_index, similarity_matrix, subject_id, task_id,  
              model=model, layer_id=hidden_layer, segmented=False, word_pos=word_pos)
    return similarity_matrix  


def make_filename_prefix(file_name_tag, subject_id, task_id, 
        model=None, segmented=False, layer_id=None, word_pos=None, output_dir=OUTPUT_DIR):
    """Generate a filename prefix based on parameters for saving/loading files."""
    file_name_parts = []
    if subject_id is not None:
        file_name_parts.append(f'subject_{subject_id}')
    if task_id is not None:
        file_name_parts.append(f'task_{task_id}')
    if word_pos is not None and len(word_pos)>0:
        pos_tag = '_'.join(word_pos)
        file_name_parts.append(f'pos_{pos_tag}')
    if segmented:
        file_name_parts.append(f'segmented')
    else: # model, and segmented.
        if model is not None:
            file_name_parts.append(f'model_{model}')
            if layer_id:
                file_name_parts.append(f'layer_{layer_id}')

    file_name_parts.append(file_name_tag)
    return f"{output_dir}/{'_'.join(file_name_parts)}"


def save_similarity_data(word_index, similarity_matrix, subject_id, task_id, 
        segmented=False, model=None, layer_id=None, word_pos=None):
    """Save both the word index and similarity matrix to files."""
    # word_index.json
    word_index_file = make_filename_prefix('word_index.json', subject_id, task_id, 
            segmented=segmented, model=model, layer_id=layer_id, word_pos=word_pos)
    # similarity_matrix
    similarity_matrix_file = make_filename_prefix('similarity_matrix.npy', subject_id, task_id, 
            segmented=segmented, model=model, layer_id=layer_id, word_pos=word_pos)
    _save_to_file(word_index, word_index_file)
    _save_to_file(similarity_matrix, similarity_matrix_file)


def sort_by_hierarchical_order(word_index, similarity_matrix):
    dissimilarity_matrix = 1 - similarity_matrix
    linkage_matrix = sch.linkage(dissimilarity_matrix, method='ward')
    dendrogram = sch.dendrogram(linkage_matrix, no_plot=True)
    order = dendrogram['leaves']
    reordered_similarity_matrix = similarity_matrix[np.ix_(order, order)]
    sorted_word_list = [word_index[i] for i in order]
    return sorted_word_list, reordered_similarity_matrix


def sort_by_spectral_clustering(word_index, similarity_matrix):
    dissimilarity_matrix = 1 - similarity_matrix
    embedding = SpectralEmbedding(n_components=1, affinity='precomputed')
    order = np.argsort(embedding.fit_transform(dissimilarity_matrix).ravel())
    reordered_similarity_matrix = similarity_matrix[np.ix_(order, order)]
    sorted_word_list = [word_index[i] for i in order]
    return sorted_word_list, reordered_similarity_matrix


def load_word_index(subject_id, task_id, 
        model=None, output_dir=OUTPUT_DIR, segmented=False, layer_id=None, word_pos=None):
    word_index_file = make_filename_prefix('word_index.json', subject_id, task_id, model=model, 
            segmented=segmented, layer_id=layer_id, word_pos=word_pos)
    return _load_from_file(word_index_file)


def load_similarity_matrix(subject_id, task_id, model=None, 
                            segmented=False, layer_id=None, word_pos=None):
    word_index = load_word_index(subject_id, task_id, 
            model=model, layer_id=layer_id, segmented=segmented, word_pos=word_pos)
    similarity_matrix_file = make_filename_prefix('similarity_matrix.npy', subject_id, task_id, 
            model=model, segmented=segmented, layer_id=layer_id, word_pos=word_pos)
    similarity_matrix = _load_from_file(similarity_matrix_file)
    return word_index, similarity_matrix

# TODO: How can we compute noise ceiling ?
#
# 1. Compute Individual RDMs
# 2. Averaging RDMs Across Subjects/Trials.
# 3. Leave-one-out Approach.
# 4. Bootstrap Resampling 
# 5. .....

def load_subject_rdms_by_task_id(task_id, word_pos=['VB']):
    pass

def compute_average_rdm(rdms): 
    f_rdm = rdms[0]
    for rdm in rdms:
        assert f_rdm.shape ==rdm.shape

    return np.mean(rdms, axis=0) 

def _get_task_rdms(task_id, word_pos=['VB']):
    rdms = []
    first_word_index = None
    for subject_id in D.load_subject_ids():
         word_index, similarity_matrix = load_similarity_matrix(subject_id, task_id, 
                 word_pos=word_pos)
         rdm = 1 - similarity_matrix
         rdms.append(rdm)
    return rdms

def _get_task_word_index(task_id, word_pos=['VB']):
     first_subject_id = D.load_subject_ids()[0]
     word_index, _ = load_similarity_matrix(first_subject_id, task_id, 
             word_pos = word_pos)
     return word_index


def compute_average_rdm_for_task_id(task_id, word_pos=['VB']):
    """
    Compute the average rdm accross participants
    """
    first_word_index = _get_task_word_index(task_id, word_pos=word_pos) 
    rdms = _get_task_rdms(task_id, word_pos=word_pos)
    return first_word_index, compute_average_rdm(rdms)


def leave_one_out_noise_ceiling(rdms):
    """
    Compute the noise ceiling using a leave-one-out approach.
    Args:
    - rdms: list of RDMs for each participant.
    
    Returns:
    - lower_bound: The lower bound of the noise ceiling.
    - upper_bound: The upper bound of the noise ceiling.
    """
    n = len(rdms)
    lower_bound_scores = []
    upper_bound_scores = []

    for i in range(n):
        # Leave-one-out average RDM excluding participant i
        loo_average_rdm = compute_average_rdm([rdms[j] for j in range(n) if j != i])

        # Correlation between participant i's RDM and LOO average
        correlation_loo = np.corrcoef(rdms[i].flatten(), loo_average_rdm.flatten())[0, 1]
        lower_bound_scores.append(correlation_loo)

        # Correlation between participant i's RDM and the overall average RDM (upper bound)
        overall_average_rdm = compute_average_rdm(rdms)
        correlation_all = np.corrcoef(rdms[i].flatten(), overall_average_rdm.flatten())[0, 1]
        upper_bound_scores.append(correlation_all)

    lower_bound = np.mean(lower_bound_scores)
    upper_bound = np.mean(upper_bound_scores)

    return lower_bound, upper_bound

def compute_noise_ceiling_bounds(task_id, word_pos=['VB']):
    first_word_index = _get_task_word_index(task_id, word_pos=word_pos) 
    rdms = _get_task_rdms(task_id, word_pos=word_pos)
    return leave_one_out_noise_ceiling(rdms)

# Determine the P-Values using permutation testing of RDMS.
def permutation_test_rsa(model_rdm, brain_rdm, n_permutations=1000):
    """
    Perform permutation testing for RSA to determine if model performance is better than chance.
    
    Args:
    - model_rdm: numpy array, the RDM of the model (e.g., BERT RDM).
    - brain_rdm: numpy array, the original RDM of brain data.
    - n_permutations: int, number of permutations to perform.
    
    Returns:
    - p_value: float, p-value indicating statistical significance of model performance.
    """
    # Compute original RSA score
    original_rsa_score = np.corrcoef(model_rdm.flatten(), brain_rdm.flatten())[0, 1]
    
    # Create null distribution by shuffling labels
    null_distribution = []
    for _ in range(n_permutations):
        # Randomly shuffle the brain RDM
        shuffled_rdm = np.random.permutation(brain_rdm.flatten()).reshape(brain_rdm.shape)
        # Compute RSA score with the shuffled RDM
        permuted_rsa_score = np.corrcoef(model_rdm.flatten(), shuffled_rdm.flatten())[0, 1]
        null_distribution.append(permuted_rsa_score)

    # Convert null distribution to a numpy array for easier manipulation
    null_distribution = np.array(null_distribution)
    
    # Calculate the p-value
    count_greater_equal = np.sum(null_distribution >= original_rsa_score)
    p_value = (count_greater_equal + 1) / (n_permutations + 1)  # Adding 1 to avoid p-value of 0

    return original_rsa_score, p_value


def compute_model_p_value(task_id, model='BERT', word_pos=['VB']): 
    # TOOD: Fix there is an issue where I am creating and  
    # saving BERT per subject which makes no sense.
    #
    # 1. Load model RDM for a subject.
    # 2. Load averge RDM across all subjects. 
    word_index, average_rdm = compute_average_rdm_for_task_id(task_id, , word_pos=word_pos)
    proto_subject_id = D.load_subject_ids()[0] # prototypical subject
    model_word_index, similarity_matrix = load_similarity_matrix(subject_id=proto_subject_id, task_id, 
            model=model, word_pos=word_pos)
    model_rdm = 1 - similarity_matrix
    assert average_rdm.shape == similarity_matrix.shape
    assert set(word_index) == model_word_index 

    original_rsa_score, p_value = permutation_test_rsa(average_rdm, model_rdm)
    return original_rsa_score, p_value


