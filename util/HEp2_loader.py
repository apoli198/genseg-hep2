import logging
import os
from os.path import splitext
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

import numpy as np
from PIL import Image
import pandas as pd

# -------------------
# Dataset for training HEp-2 cell segmentation
# -------------------
class TrainingDataset(Dataset):
    def __init__(self, data_dir: str, scale: float = 1.0):
        self.data_dir = Path(data_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale

        num_intermediate = None     # 5 | 25 | 50 | None
        num_positive = None         # 5 | 25 | 50 | None
        random_state = 42        # 1, 2, 3, 4, 5 | 10, 20, 30 | 100, 200 | 42

        full_df = pd.read_csv(os.path.join(data_dir, "train_folds.csv"))
        if full_df.empty:
            raise RuntimeError(f'No input file found in {data_dir}, make sure you put your images there')
        
        if num_intermediate is not None and num_positive is not None:
            sampled_patients = self._sample_patients(
                full_df, num_intermediate, num_positive, random_state
            )
            self.df = full_df[full_df['patient_id'].isin(sampled_patients)].reset_index(drop=True)
        else:
            self.df = full_df
        
        logging.info(f'Creating dataset with {len(self.df)} examples')

    def _sample_patients(self, df, num_intermediate, num_positive, random_state):
        """
        Sample patients with specified intensities evenly across all folds.
        
        Parameters:
            df: Full dataframe with patient data
            num_intermediate: Number of intermediate patients to sample
            num_positive: Number of positive patients to sample
            random_state: Random seed for sampling
            
        Returns:
            set: Set of sampled patient IDs
        """
        grp = df.groupby('patient_id')
        patient_fold = grp['fold'].first()
        patient_int = grp['intensity'].first()
        unique_folds = np.sort(patient_fold.unique())
        num_folds = len(unique_folds)
        rng = np.random.RandomState(random_state)

        def sample_across_folds(intensity, total):
            """Sample patients with specific intensity across folds."""
            if total is None:
                return patient_int[patient_int == intensity].index.values
            if total % num_folds != 0:
                raise ValueError(f"num_{intensity} must be divisible by {num_folds}")
            
            per_fold = total // num_folds
            sampled = []
            for f in unique_folds:
                pats = patient_fold[patient_fold == f].index
                pats = pats[patient_int.loc[pats] == intensity]
                if len(pats) < per_fold:
                    raise ValueError(
                        f"Not enough {intensity} patients in fold {f}: "
                        f"requested {per_fold}, available {len(pats)}"
                    )
                sampled.extend(rng.choice(pats, size=per_fold, replace=False))
            return np.array(sampled)
        
        inter_pats = sample_across_folds('intermediate', num_intermediate)
        pos_pats = sample_across_folds('positive', num_positive)
        return set(np.concatenate([inter_pats, pos_pats]))

    def __len__(self):
        return len(self.df)

    @staticmethod
    def preprocess(pil_img, scale, is_mask):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img_ndarray = np.asarray(pil_img)

        if not is_mask:
            if img_ndarray.ndim == 2:
                img_ndarray = img_ndarray[np.newaxis, ...]
            else:
                img_ndarray = img_ndarray.transpose((2, 0, 1))
            img_ndarray = img_ndarray / 255.0
            return img_ndarray
        else:
            img_ndarray = (img_ndarray > 127).astype(np.float32)
            img_ndarray = np.expand_dims(img_ndarray, axis=0)
            return img_ndarray
    
    @staticmethod
    def preprocess_pix2pix(pil_img, scale, is_mask):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img_ndarray = np.asarray(pil_img)

        if not is_mask:
            if img_ndarray.ndim == 2:
                img_ndarray = img_ndarray[np.newaxis, ...]
            else:
                img_ndarray = img_ndarray.transpose((2, 0, 1))

            img_ndarray = ((img_ndarray / 255.0) * 2.0) - 1.0
            return img_ndarray
        else:
            img_ndarray = (img_ndarray > 127).astype(np.float32)
            img_ndarray = np.expand_dims(img_ndarray, axis=0)
            return img_ndarray

    @staticmethod
    def load(filename):
        ext = splitext(filename)[1]
        if ext == '.npy':
            return Image.fromarray(np.load(filename))
        elif ext in ['.pt', '.pth']:
            return Image.fromarray(torch.load(filename).numpy())
        else:
            return Image.open(filename)
    
    @staticmethod
    def load_pix2pix(filename):
        ext = splitext(filename)[1]
        if ext == '.npy':
            return Image.fromarray(np.load(filename))
        elif ext in ['.pt', '.pth']:
            return Image.fromarray(torch.load(filename).numpy())
        else:
            return Image.open(filename)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filename = row["filename"]
        img_file = os.path.join(self.data_dir, "train_patch", f"{filename}.tif")
        mask_file = os.path.join(self.data_dir, "train_patch", f"{filename}_Mask.tif")
        mask = self.load(mask_file).convert('L')
        img = self.load(img_file).convert('L')
        assert img.size == mask.size, \
          f'Image and mask {filename} should be the same size, but are {img.size} and {mask.size}'
        img = self.preprocess(img, self.scale, is_mask=False)
        mask = self.preprocess(mask, self.scale, is_mask=True)

        mask_pix2pix = self.load_pix2pix(mask_file).convert('L')
        img_pix2pix = self.load_pix2pix(img_file).convert('L')
        assert img_pix2pix.size == mask_pix2pix.size, \
           f'Image and mask {filename} should be the same size, but are {img_pix2pix.size} and {mask_pix2pix.size}'

        img_pix2pix = self.preprocess_pix2pix(img_pix2pix, self.scale, is_mask=False)
        mask_pix2pix = self.preprocess_pix2pix(mask_pix2pix, self.scale, is_mask=True)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).float().contiguous(),
            'image_pix2pix': torch.as_tensor(img_pix2pix.copy()).float().contiguous(),
            'mask_pix2pix': torch.as_tensor(mask_pix2pix.copy()).float().contiguous()
        }
    
# -------------------
# Dataset for testing HEp-2 cell segmentation
# -------------------
class TestDataset(Dataset):
    def __init__(self, data_dir: str, scale: float = 1.0):
        self.data_dir = Path(data_dir)
        self.scale = scale
        self.patch_size = 256

        # Create a dataframe of unique whole images, not patches
        test_df = pd.read_csv(os.path.join(data_dir, "test_indices.csv"))
        # Get unique base filenames (e.g., '00001_p0')
        test_df['base_filename'] = test_df['filename'].apply(lambda x: '_'.join(x.split('_')[:2]))
        self.unique_images = test_df['base_filename'].unique().tolist()
        
        logging.info(f'Creating dataset with {len(self.unique_images)} whole images.')

    def __len__(self):
        return len(self.unique_images)

    def __getitem__(self, idx):
        # Get the base name for one whole image (e.g., '00001_p0')
        base_name = self.unique_images[idx]
        
        # --- Load all 30 patches for this image ---
        patch_files = [os.path.join(self.data_dir, "test_patch", f"{base_name}_patch{i}.tif") for i in range(30)]
        # Use the static methods from TrainingDataset to load and preprocess
        patches = [TrainingDataset.load(p).convert('L') for p in patch_files]
        processed_patches = [TrainingDataset.preprocess(p, self.scale, is_mask=False) for p in patches]

        # --- Load the single ground truth mask ---
        whole_mask_file = os.path.join(self.data_dir, "test", f"{base_name}_Mask.tif")
        whole_mask = TrainingDataset.load(whole_mask_file).convert('L')
        # Use the static methods from TrainingDataset
        whole_mask = TrainingDataset.preprocess(whole_mask, self.scale, is_mask=True)

        return {
            'patches': torch.as_tensor(np.array(processed_patches)).float().contiguous(),       # Shape: (30, 1, 256, 256)
            'whole_mask': torch.as_tensor(whole_mask.copy()).float().contiguous()                # Shape: (1, 1040, 1388)
        }
