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
from . import dataset as D

def _get_similarity_matrix(subject_id='01', session_id=0, task_id=0):
    word_metadata_df, word_epoch_map = D._get_epoch_word_map(subject_id, session_id, task_id)

	# Initialize dictionary to store ICA-transformed epochs
	ica_epochs = {}

	# Loop through each target word and apply ICA
	for word, epochs in target_word_epochs.items():
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

	# Extract ICA data for RSA
	target_word_vectors = []

	for word, epochs_ica in ica_epochs.items():
		# Average the ICA components over time
		avg_ica = epochs_ica.average().get_data()  # Shape: (n_channels, n_times)

		# Flatten the data (optional: you can decide to not flatten depending on your approach)
		vector = avg_ica.flatten()
		target_word_vectors.append(vector)

	# Convert to numpy array
	target_word_vectors = np.array(target_word_vectors)

	# Compute cosine similarity matrix
	similarity_matrix = cosine_similarity(target_word_vectors)

    return similarity_matrix
