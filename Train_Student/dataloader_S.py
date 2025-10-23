import sys
sys.path.append(r"/data/Desktop/Semi-NC") 

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Tuple, Dict, Any
import random
import cv2
import os
from augmentation import *
from utils import *
from Train_Teacher.Course import Course_Augmentation
import torchvision.transforms as transforms
from config import *
args = Params() #all files only have one args，all files can change it，but only this one！
 
transform_tensor = transforms.ToTensor()

class SemiSupervisedDataset(Dataset):
    def __init__(self, args, mode="train"):
        self.mode = mode
        self.args = args
        
        self.images, self.nerve_labels, self.cell_labels, self.images_name = self.read_datasets(mode, args)
        
        self.labeled_indices = []
        self.unlabeled_indices = []
        
        for idx in range(len(self.images)):
            nerve_path = self.nerve_labels[idx]
            cell_path = self.cell_labels[idx]
            
            nerve_label = cv2.imread(nerve_path, 0)
            cell_label = cv2.imread(cell_path, 0)
            
            if self.is_blank_mask(nerve_label) and self.is_blank_mask(cell_label):
                self.unlabeled_indices.append(idx)
            else:
                self.labeled_indices.append(idx)
        
        self.epoch_finished = False
        self.current_index = 0

    def read_datasets(self, mode, args):
       
        images = []
        cell_labels = []
        nerve_labels = []
        images_name = []

        if mode == "train":
            train_folder = os.path.join(args.data_path, 'train')
            image_folder = os.path.join(train_folder, 'image')
            images_name = os.listdir(image_folder)
            gt3_cell_folder = os.path.join(train_folder, 'cell_label')
            gt3_nerve_folder = os.path.join(train_folder, 'nerve_label')
             
        elif mode == "val":
            val_folder = os.path.join(args.data_path, "val")
            image_folder = os.path.join(val_folder, 'image')
            images_name = os.listdir(image_folder)
            gt3_cell_folder = os.path.join(val_folder, 'cell_label')
            gt3_nerve_folder = os.path.join(val_folder, 'nerve_label')

        elif mode == "test":
            test_folder = os.path.join(args.data_path, "test")
            image_folder = os.path.join(test_folder, 'image')
            images_name = os.listdir(image_folder)
            gt3_cell_folder = os.path.join(test_folder, 'cell_label')
            gt3_nerve_folder = os.path.join(test_folder, 'nerve_label')

        for name in images_name:
            img_path = os.path.join(image_folder, name)
            gt3_cell_path = os.path.join(gt3_cell_folder, name)
            gt3_nerve_path = os.path.join(gt3_nerve_folder, name)

            images.append(img_path)
            cell_labels.append(gt3_cell_path)
            nerve_labels.append(gt3_nerve_path)
         
        return images, nerve_labels, cell_labels, images_name

    def is_blank_mask(self, mask):
        if mask is None:
            return True
        return np.all(mask == 0)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if idx < len(self.labeled_indices):
            self.epoch_finished = False
            is_labeled = True
            original_idx = self.labeled_indices[idx]
        else:
            self.epoch_finished = True
            is_labeled = False
            unlabeled_idx = idx - len(self.labeled_indices)
            original_idx = self.unlabeled_indices[unlabeled_idx]
        
        self.current_index = idx
        
        image_path = self.images[original_idx]
        nerve_label_path = self.nerve_labels[original_idx]
        cell_label_path = self.cell_labels[original_idx]
        image_name = self.images_name[original_idx]

        image = cv2.imread(image_path)
        nerve_label = cv2.imread(nerve_label_path, 0)
        cell_label = cv2.imread(cell_label_path, 0)

        if len(image.shape) == 2:
            image = image[:, :, np.newaxis]
            image = np.repeat(image, 3, axis=-1)

        nerve_label = nerve_label[:, :, np.newaxis]
        cell_label = cell_label[:, :, np.newaxis]

        if self.args.crop == True and self.mode == "train":
            image, nerve_label, cell_label = crop_images_and_label(
                image, nerve_label, cell_label, self.args.roi_size
            )

        if self.args.data_path_selection != "teacher" and self.mode == self.args.enhance_mode_S:
            image, nerve_label, cell_label = apply_augmentations_KD(image, nerve_label, cell_label)

        if not is_labeled:
            nerve_pseudo_label_folder = os.path.join(self.args.data_path, 'train', 'nerve_pseudo_label')
            cell_pseudo_label_folder = os.path.join(self.args.data_path, 'train', 'cell_pseudo_label')
            nerve_pseudo_label_path = os.path.join(nerve_pseudo_label_folder, image_name)
            cell_pseudo_label_path = os.path.join(cell_pseudo_label_folder, image_name)
            
            if os.path.exists(nerve_pseudo_label_path) and os.path.exists(cell_pseudo_label_path):
                nerve_pseudo_label = cv2.imread(nerve_pseudo_label_path, 0)
                cell_pseudo_label = cv2.imread(cell_pseudo_label_path, 0) 
                pseudo_label = cv2.add(nerve_pseudo_label, cell_pseudo_label)
                pseudo_label = pseudo_label[:, :, np.newaxis]
                SDF_image,_ = get_SDF_data(image, pseudo_label, self.args.beta) 
            else:
                SDF_image = image

        else:
            label = cv2.add(nerve_label, cell_label)
            SDF_image, _ = get_SDF_data(image, label, self.args.beta)

        image_tensor = transform_tensor(image)
        SDF_image_tensor = transform_tensor(SDF_image)
        nerve_label_tensor = transform_tensor(nerve_label)
        cell_label_tensor = transform_tensor(cell_label)

        sample = {
            'image': image_tensor,
            'SDF_image': SDF_image_tensor,
            'nerve_label': nerve_label_tensor,
            'cell_label': cell_label_tensor,
            'image_name': image_name,
            'is_labeled': is_labeled,
            'epoch_finished': self.epoch_finished,
            'current_index': idx
        }

        return sample

    def get_epoch_status(self):
        return {
            'epoch_finished': self.epoch_finished,
            'current_index': self.current_index,
            'total_labeled': len(self.labeled_indices),
            'total_unlabeled': len(self.unlabeled_indices),
            'current_phase': 'unlabeled' if self.epoch_finished else 'labeled'
        }

class SemiSupervisedDataLoader:
    def __init__(self, dataset: SemiSupervisedDataset, batch_size: int = 8, 
                 shuffle: bool = True, num_workers: int = 4):

        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        
        self.labeled_loader = self._create_labeled_loader()
        
        self.unlabeled_loader = self._create_unlabeled_loader() if len(self.dataset.unlabeled_indices) > 0 else None
        
        self.current_loader = self.labeled_loader
        self.labeled_exhausted = False
        self.has_unlabeled_data = len(self.dataset.unlabeled_indices) > 0
        
    def _create_labeled_loader(self):
        labeled_dataset = torch.utils.data.Subset(
            self.dataset, 
            range(len(self.dataset.labeled_indices))
        )
        return DataLoader(
            labeled_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=False,
            # drop_last=True
        )
    
    def _create_unlabeled_loader(self):
        if len(self.dataset.unlabeled_indices) == 0:
            return None
            
        unlabeled_indices = list(range(
            len(self.dataset.labeled_indices), 
            len(self.dataset)
        ))
        unlabeled_dataset = torch.utils.data.Subset(
            self.dataset, 
            unlabeled_indices
        )
        return DataLoader(
            unlabeled_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=False,
            # drop_last=True
        )
    
    def __iter__(self):
        self.labeled_exhausted = False
        self.current_loader = self.labeled_loader
        
        self.labeled_iterator = iter(self.labeled_loader)
        self.unlabeled_iterator = iter(self.unlabeled_loader) if self.has_unlabeled_data else None
        
        return self
    
    def __next__(self):
        try:
            if not self.labeled_exhausted:
                try:
                    batch = next(self.labeled_iterator)
                    batch['phase'] = 'labeled'
                    return batch
                except StopIteration:
          
                    self.labeled_exhausted = True
                   
                    if not self.has_unlabeled_data:
                        raise StopIteration
                    
            if self.has_unlabeled_data:
                batch = next(self.unlabeled_iterator)
                batch['phase'] = 'unlabeled'
                return batch
            else:
                raise StopIteration
            
        except StopIteration:
           
            raise StopIteration
    
    def __len__(self):
        labeled_len = len(self.labeled_loader)
        unlabeled_len = len(self.unlabeled_loader) if self.has_unlabeled_data else 0
        return labeled_len + unlabeled_len


class SemiSupervisedDataLoaderFactory:
    def __init__(self):
        pass
    
    def load_train_data(self, args, batch_size):
        dataset = SemiSupervisedDataset(args, mode="train")
        train_loader = SemiSupervisedDataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4
        )
        return train_loader
    
    def load_val_data(self, args, batch_size):
        dataset = SemiSupervisedDataset(args, mode="val")
        val_loader = SemiSupervisedDataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,  
            num_workers=4
        )
        return val_loader
    
    def load_test_data(self, args, batch_size):
        dataset = SemiSupervisedDataset(args, mode="test")
        test_loader = SemiSupervisedDataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False, 
            num_workers=4
        )
        return test_loader

def create_unlabeled_loader(args, batch_size=4):

    dataset = SemiSupervisedDataset(args, mode="train")
    
    unlabeled_indices = list(range(len(dataset.labeled_indices), len(dataset)))
    
    unlabeled_dataset = torch.utils.data.Subset(dataset, unlabeled_indices)
    
    unlabeled_loader = DataLoader(
        unlabeled_dataset,
        batch_size=batch_size,
        shuffle=False, 
        num_workers=4,
        pin_memory=False
    )
    
    return unlabeled_loader
 
