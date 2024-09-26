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
    D._get_epoch_word_map(subject_id, session_id, task_id)
    return word_epochs
#
