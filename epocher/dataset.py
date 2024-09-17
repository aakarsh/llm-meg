

import numpy as np
import mne
from mne_bids import BIDSPath # Import the BIDSPath class

# Create a BIDSPath object


DATASET_ROOT="/content/drive/MyDrive/TUE-SUMMER-2024/ulm-meg/"

def load_bids_path(root=DATASET_ROOT, subject="sub-01", datatype="meg", session="0"):
    bids_path = BIDSPath(root=DATASET_ROOT+"/"+subject, session=session, datatype=datatype)
    return bids_path

