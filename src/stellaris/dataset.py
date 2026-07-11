from pathlib import Path
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import GroupShuffleSplit

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "kepler_trainset.h5"

class StellarisDataset(Dataset):
    """
    Loads (global_view, local_view, label) triples from the HDF5 file.
    """

    def __init__(self, hdf5_path: Path, indices: np.ndarray):
        self.hdf5_path = Path(hdf5_path)
        self.indices = np.asarray(indices, dtype=np.int64)

        with h5py.File(self.hdf5_path, 'r') as f:
            self.global_views = np.asarray(f['global_view'][:], dtype=np.float32)
            self.local_views = np.asarray(f['local_view'][:], dtype=np.float32)
            self.labels = np.asarray(f['label'][:], dtype=np.float32)
            self.kepids = np.asarray(f['kepid'][:], dtype=np.int64)
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> dict:
        i = self.indices[idx]
        
        return {
            'global_view': torch.from_numpy(self.global_views[i]).unsqueeze(0),
            'local_view': torch.from_numpy(self.local_views[i]).unsqueeze(0),
            'label': torch.tensor(self.labels[i], dtype=torch.float32),
        }

def make_splits(
    hdf5_path: Path = DEFAULT_DATASET_PATH,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    # From the Hitchhiker's Guide to the Galaxy :)
    # If you know this, you know ball 
    seed: int = 42,
) -> dict:
    """
    Build the train/validation/test index splits such that there
    is no data leakage between them.


    Returns a dict with keys 'train', 'val', 'test', each an ndarray of indices.
    """

    with h5py.File(hdf5_path, 'r') as f:
        kepids = np.asarray(f['kepid'][:])
        n = len(kepids)
    
    indices = np.arange(n)

    # Split test set
    gss = GroupShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
    train_val_idx, test_idx = next(gss.split(indices, groups=kepids))

    # Split validation set from the rest
    relative_val_frac = val_frac / (1.0 - test_frac)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=relative_val_frac, random_state=seed)
    train_idx_rel, val_idx_rel = next(
        gss2.split(train_val_idx, groups=kepids[train_val_idx])
    )

    train_idx = train_val_idx[train_idx_rel]
    val_idx = train_val_idx[val_idx_rel]

    # Verifying no Kepler ID appears in more than one split
    train_kepids = set(kepids[train_idx].tolist())
    val_kepids = set(kepids[val_idx].tolist())
    test_kepids = set(kepids[test_idx].tolist())
    assert not (train_kepids & val_kepids), "KepID leak between train and val"
    assert not (train_kepids & test_kepids), "KepID leak between train and test"
    assert not (val_kepids & test_kepids), "KepID leak between val and test"

    return {
        'train': np.sort(train_idx),
        'val': np.sort(val_idx),
        'test': np.sort(test_idx),
    }    



