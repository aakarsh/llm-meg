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
from sklearn.preprocessing import normalize

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

# Function to save cache
def _save_cache(filename, data):
    """Save data to cache in either JSON or NumPy format."""
    if filename.endswith('.json'):
        with open(filename, 'w') as f:
            json.dump(data, f)
    elif filename.endswith('.npy'):
        np.save(filename, data)
    elif filename.endswith('.npz'):
        if isinstance(data, dict):
            np.savez(filename, **{key: value for key, value in data.items()})
        else:
            raise ValueError("Data must be a dictionary to save in '.npz' format.")
    else:
        raise ValueError("Unsupported file format. Use .json or .npy")

# Function to load cache
def _load_cache(filename):
    """Load data from cache in either JSON or NumPy format."""
    if filename.endswith('.json'):
        with open(filename, 'r') as f:
            return json.load(f)
    elif filename.endswith('.npy'):
        return np.load(filename, allow_pickle=True).item()
    elif filename.endswith('.npz'):
        data = np.load(filename, allow_pickle=True)
        return {key: data[key] for key in data.files}
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
    """
    # TOOD: Fix there is an issue where I am creating and  
    # saving BERT per subject which makes no sense.
    #
    # 1. Load model RDM for a subject.
    # 2. Load averge RDM across all subjects. 
    """
    word_index, average_rdm = compute_average_rdm_for_task_id(task_id, word_pos=word_pos)
    proto_subject_id = D.load_subject_ids()[0] # prototypical subject
    model_word_index, similarity_matrix = load_similarity_matrix(proto_subject_id, task_id, 
            model=model, word_pos=word_pos)

    model_rdm = 1 - similarity_matrix

    assert average_rdm.shape == similarity_matrix.shape
    assert set(word_index) == set(model_word_index)

    original_rsa_score, p_value = permutation_test_rsa(average_rdm, model_rdm)
    return original_rsa_score, p_value

def sliding_window_rsa_per_electrode(subject_id='01', session_id=0, task_id=0, 
        window_size=0.05, step_size=0.01, word_pos=['VB'], use_ica=False, cache_output=True):
    """
    Perform RSA between words for each electrode and each time window using a sliding window approach.
    
    Args:
    - subject_id: Subject identifier.
    - session_id: Session identifier.
    - task_id: Task identifier.
    - window_size: Size of the sliding window in seconds.
    - step_size: Step size for the sliding window in seconds.
    - word_pos: List of parts of speech tags to consider.
    - use_ica: Boolean indicating if ICA-transformed data should be used.
    
    Returns:
    - rsa_matrices_per_electrode: A dictionary with electrode names as keys and lists of RSA matrices 
    (n_words x n_words) for each sliding window.
    - time_points: A list of time points for each window.
    """

    cache_file = make_filename_prefix("sliding_window_rsa.npz", subject_id, task_id)

    # Load from cache if available
    if cache_output and os.path.exists(cache_file):
        loaded_data = _load_cache(cache_file)
        rsa_matrices = loaded_data['rsa_matrices']
        channel_names = loaded_data['channel_names']
        time_points = loaded_data['time_points']
        pos = loaded_data['pos']
        # Reconstruct the dictionary format for returning
        rsa_matrices_per_electrode = {channel_names[i]: rsa_matrices[:, i] for i in range(len(channel_names))}
        return rsa_matrices_per_electrode, time_points, pos

    # Load word epochs for each word using your existing method
    word_index, word_epoch_map = D._load_epoch_map(subject_id, session_id, task_id, 
            use_ica=use_ica, word_pos=word_pos)

    # Initialize the sliding window RSA
    sfreq = word_epoch_map[word_index[0]].info['sfreq']  # Sampling frequency from the epochs
    window_samples = int(window_size * sfreq)
    step_samples = int(step_size * sfreq)

    # Get the number of electrodes from one of the epochs
    n_channels = word_epoch_map[word_index[0]].info['nchan']
    channel_names = word_epoch_map[word_index[0]].info['ch_names']
    info = word_epoch_map[word_index[0]].info
    pos = mne.find_layout(info).pos # electrode potentials

    rsa_matrices = []  # Initialize list to store RSA matrices for all channels
    time_points = []

    # Perform sliding window analysis
    for start in range(0, len(word_epoch_map[word_index[0]].times) - window_samples + 1, step_samples):
        end = start + window_samples

        rsa_matrices_window = []
        # Loop through each electrode
        for ch_idx in range(n_channels):
            # Initialize a list to store activation for each word for the current electrode and time window
            word_vectors = []

            # Loop through each word
            for word in word_index:
                epochs = word_epoch_map[word]  # Shape: (n_epochs, n_channels, n_times)

                if len(epochs) == 0:
                    continue

                # Extract the electrode data for the current time window
                epoch_window = epochs.get_data()[:, ch_idx, start:end]  # Shape: (n_epochs, window_samples)
                avg_window = np.mean(epoch_window, axis=0)  # Average over epochs, Shape: (window_samples,)

                # Normalize the average window vector
                avg_vector = avg_window / (np.linalg.norm(avg_window) + 1e-10)  # Normalized vector, Shape: (window_samples,)
                word_vectors.append(avg_vector)

            # Convert to numpy array and normalize all word vectors
            word_vectors = np.array(word_vectors)  # Shape: (n_words, window_samples)
            word_vectors = normalize(word_vectors, axis=1)  # Normalize each word vector to unit length

            # Calculate cosine similarity between all words (i.e., the RSA matrix)
            rsa_matrix = cosine_similarity(word_vectors)  # Shape: (n_words, n_words)

            # Store the RSA matrix for this electrode and time window
            rsa_matrices_window.append(rsa_matrix)

        # Append the RSA matrices for this window to the main list
        rsa_matrices.append(rsa_matrices_window)
        # Store the midpoint of the current time window
        time_points.append(word_epoch_map[word_index[0]].times[start] + (window_size / 2))

    # Convert rsa_matrices to numpy array for saving
    rsa_matrices = np.array(rsa_matrices)  # Shape: (n_windows, n_channels, n_words, n_words)

    if cache_output:
        _save_cache(cache_file, { 'rsa_matrices': rsa_matrices,
                                  'channel_names': channel_names, 
                                  'pos': pos, # Q: Will it save correctly ?
                                  'time_points': time_points })
        print(f"Saved computed results to cache at {cache_file}")

    rsa_matrices_per_electrode = { channel_names[i]: rsa_matrices[:, i] for i in range(len(channel_names))}

    return rsa_matrices_per_electrode, time_points, pos

def plot_rsa_lineplot_over_time(subject_id, task_id, session_id=0, model='BERT',
                                window_size=0.05, step_size=0.01, word_pos=['VB'],
                                use_ica=False, cache_output=True):
    """
    Plot RSA line plot over time to visualize alignment with BERT model.

    Args:
    - subject_id, task_id, session_id: identifiers for MEG data.
    - model: The model to compare the MEG signals with (e.g., BERT).
    - window_size, step_size: parameters for the sliding window RSA.
    - word_pos: The parts of speech tags for filtering words.
    - use_ica: Boolean indicating if ICA-transformed data should be used.
    - cache_output: Whether to use cached computations.
    """

    # Perform RSA per electrode using sliding window
    rsa_matrices_per_electrode, time_points, pos = sliding_window_rsa_per_electrode(
        subject_id=subject_id,
        session_id=session_id,
        task_id=task_id,
        window_size=window_size,
        step_size=step_size,
        word_pos=word_pos,
        use_ica=use_ica,
        cache_output=cache_output)

    # Load the BERT model RDM for comparison
    proto_subject_id = D.load_subject_ids()[0]  # Prototypical subject for BERT
    _, bert_similarity_matrix = load_similarity_matrix(proto_subject_id, task_id, 
            model=model, word_pos=word_pos)
    bert_rdm = 1 - bert_similarity_matrix # Convert similarity matrix to RDM

    # Prepare to collect RSA alignment scores for each time window across electrodes
    ch_names = list(rsa_matrices_per_electrode.keys())
    rsa_scores_per_window = []

    for t_idx, time_point in enumerate(time_points):
        rsa_scores = []
        for ch_name in ch_names:
            # Extract the RSA matrix for the given channel at the current time window
            rsa_matrix = rsa_matrices_per_electrode[ch_name][t_idx]

            # Calculate RSA alignment score between electrode RDM and BERT RDM
            rsa_score = np.corrcoef(rsa_matrix.flatten(), bert_rdm.flatten())[0, 1]
            rsa_scores.append(rsa_score)

        rsa_scores_per_window.append(rsa_scores)

    # Convert to numpy array: Shape (n_windows, n_channels)
    rsa_scores_per_window = np.array(rsa_scores_per_window)

    # Average across electrodes for each time point
    avg_rsa_scores_over_time = np.mean(rsa_scores_per_window, axis=1)

    # Plot the average RSA scores over time
    plt.figure(figsize=(10, 6))
    plt.plot(time_points, avg_rsa_scores_over_time, marker='o', linestyle='-', color='b')
    plt.xlabel('Time (s)')
    plt.ylabel('Average RSA Score')
    plt.title('Average RSA Alignment Over Time with BERT Model')
    plt.grid(True)
    plt.savefig(f'{IMAGES_DIR}/subject-{subject_id}-task-{task_id}-rsa_lineplot.png')
    plt.show()


def plot_rsa_lineplot_per_channel(subject_id, task_id, session_id=0, model='BERT',
                                  window_size=0.05, step_size=0.01, word_pos=['VB'],
                                  use_ica=False, cache_output=True):
    """
    Plot RSA line plot per electrode over time to visualize alignment with BERT model.

    Args:
    - subject_id, task_id, session_id: identifiers for MEG data.
    - model: The model to compare the MEG signals with (e.g., BERT).
    - window_size, step_size: parameters for the sliding window RSA.
    - word_pos: The parts of speech tags for filtering words.
    - use_ica: Boolean indicating if ICA-transformed data should be used.
    - cache_output: Whether to use cached computations.
    """

    # Perform RSA per electrode using sliding window
    rsa_matrices_per_electrode, time_points, pos = sliding_window_rsa_per_electrode(
        subject_id=subject_id,
        session_id=session_id,
        task_id=task_id,
        window_size=window_size,
        step_size=step_size,
        word_pos=word_pos,
        use_ica=use_ica,
        cache_output=cache_output
    )

    # Load the BERT model RDM for comparison
    proto_subject_id = D.load_subject_ids()[0]  # Prototypical subject for BERT
    _, bert_similarity_matrix = load_similarity_matrix(proto_subject_id, task_id, model=model, word_pos=word_pos)
    bert_rdm = 1 - bert_similarity_matrix  # Convert similarity matrix to RDM

    # Prepare to collect RSA alignment scores for each time window across electrodes
    ch_names = list(rsa_matrices_per_electrode.keys())
    rsa_scores_per_window = []

    for t_idx, time_point in enumerate(time_points):
        rsa_scores = []
        for ch_name in ch_names:
            # Extract the RSA matrix for the given channel at the current time window
            rsa_matrix = rsa_matrices_per_electrode[ch_name][t_idx]

            # Calculate RSA alignment score between electrode RDM and BERT RDM
            rsa_score = np.corrcoef(rsa_matrix.flatten(), bert_rdm.flatten())[0, 1]
            rsa_scores.append(rsa_score)

        rsa_scores_per_window.append(rsa_scores)

    # Convert to numpy array: Shape (n_windows, n_channels)
    rsa_scores_per_window = np.array(rsa_scores_per_window).T

    # Plot the RSA scores over time for each channel
    plt.figure(figsize=(14, 10))
    for ch_idx, ch_name in enumerate(ch_names):
        plt.plot(time_points, rsa_scores_per_window[ch_idx], label=f'Channel {ch_name}')

    plt.xlabel('Time (s)')
    plt.ylabel('RSA Score')
    plt.title('RSA Scores Over Time for Each Electrode (Aligned with BERT Model)')
    plt.legend(loc='best', ncol=2, fontsize='small')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{IMAGES_DIR}/subject-{subject_id}-task-{task_id}-rsa_per_channel_lineplot.png')
    plt.show()

def plot_rsa_topomap_over_time(subject_id, task_id, session_id=0, model='BERT', 
                               window_size=0.05, step_size=0.01, word_pos=['VB'], 
                               use_ica=False, cache_output=True):
    """
    Plot RSA topomap over time to visualize alignment with BERT model.

    Args:
    - subject_id, task_id, session_id: identifiers for MEG data.
    - model: The model to compare the MEG signals with (e.g., BERT).
    - window_size, step_size: parameters for the sliding window RSA.
    - word_pos: The parts of speech tags for filtering words.
    - use_ica: Boolean indicating if ICA-transformed data should be used.
    - cache_output: Whether to use cached computations.
    """
    # Perform RSA per electrode using sliding window
    rsa_matrices_per_electrode, time_points, pos = sliding_window_rsa_per_electrode(
        subject_id=subject_id,
        session_id=session_id,
        task_id=task_id,
        window_size=window_size,
        step_size=step_size,
        word_pos=word_pos,
        use_ica=use_ica,
        cache_output=cache_output
    )

    # Load the BERT model RDM for comparison
    proto_subject_id = D.load_subject_ids()[0]  # Prototypical subject for BERT
    _, bert_similarity_matrix = load_similarity_matrix(proto_subject_id, task_id, model=model, word_pos=word_pos)
    bert_rdm = 1 - bert_similarity_matrix  # Convert similarity matrix to RDM

    # Prepare to collect RSA alignment scores for each time window across electrodes
    ch_names = list(rsa_matrices_per_electrode.keys())
    rsa_scores_per_window = []

    for t_idx, time_point in enumerate(time_points):
        rsa_scores = []
        for ch_name in ch_names:
            # Extract the RSA matrix for the given channel at the current time window
            rsa_matrix = rsa_matrices_per_electrode[ch_name][t_idx]

            # Calculate RSA alignment score between electrode RDM and BERT RDM
            rsa_score = np.corrcoef(rsa_matrix.flatten(), bert_rdm.flatten())[0, 1]
            rsa_scores.append(rsa_score)

        rsa_scores_per_window.append(rsa_scores)

    rsa_scores_per_window = np.array(rsa_scores_per_window).T  # Shape: (n_channels, n_windows)


    # Create MNE info with MEG channel types

    # Use MEG-specific layout for plotting
    #info.set_montage(montage)  # Set the montage separately

    for t_idx, time_point in enumerate(time_points):
        plt.figure(figsize=(160, 150))
        im, _ = mne.viz.plot_topomap(rsa_scores_per_window[:, t_idx], pos, show=False, names=ch_names)
        plt.title(f'RSA Topomap at Time: {time_point:.2f} s')
        plt.colorbar(im)  # Use the mappable object returned by plot_topomap
        plt.savefig(f'{IMAGES_DIR}/subject-{subject_id}-task-{task_id}-rsa_topomap_{t_idx:02d}.png')
        plt.close()
