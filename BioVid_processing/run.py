import os

from .ECG.train import main as ecg_train
from .GSR.train import main as gsr_train

npz_dirs = {
    "ecg":"./processed/ecg",
    "gsr":"./processed/gsr"
}


def model_run(model_name = "", mode = ""):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if model_name == 'ecg' and mode == 'train':
        npz_dir = npz_dirs['ecg']
        abs_npz_dir = os.path.abspath(os.path.join(base_dir, npz_dir))
        ecg_train(data_dir = abs_npz_dir)
    elif model_name == 'gsr' and mode == 'train':
        npz_dir = npz_dirs['gsr']
        abs_npz_dir = os.path.abspath(os.path.join(base_dir, npz_dir))
        gsr_train(data_dir = abs_npz_dir)
    else:
        raise ValueError('Model name must be ecg or gsr')