# -- coding: utf-8 --
import torch
import os
import shutil
from copy import deepcopy
import shutil
import numpy as np
import cv2
from scipy.ndimage import distance_transform_edt as distance
from skimage import segmentation as skimage_seg
import numpy as np
from scipy.ndimage import binary_fill_holes
import torch
import torch.nn.functional as F
import glob


def load_model_weights_only(model, checkpoint_path, args=None):

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"No existence: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint)
    
    filename = os.path.basename(checkpoint_path)
    try:
        
        epoch_num = int(filename.split('_')[-1].split('.')[0])
        start_epoch = epoch_num + 1 
    except:
       
        start_epoch = 1
    
    return model, start_epoch

def delete_previous_checkpoints(folder_path, keep_current=None):

    if not os.path.exists(folder_path):
        return
        
    pattern = os.path.join(folder_path, "*")
    all_files = glob.glob(pattern)
    model_files = [f for f in all_files if f.endswith('.pkl') or f.endswith('.pth')]
    
    if not model_files:
        return
    
    model_files.sort(key=os.path.getmtime)
    
    if keep_current and keep_current in model_files:
        files_to_delete = [f for f in model_files if f != keep_current]
    else:
        files_to_delete = model_files[:-1] if len(model_files) > 1 else []
    
    for file_path in files_to_delete:
        try:
            os.remove(file_path)
            print(f"Delete: {os.path.basename(file_path)}")
        except Exception as e:
            print(f"Delete failure {file_path}: {e}")
    
    remaining_files = [f for f in model_files if f not in files_to_delete]
    if remaining_files:
        print(f"{[os.path.basename(f) for f in remaining_files]}")

def generate_pseudo_labels(args, device, model, unlabeled_loader, epoch):

    if torch.is_tensor(epoch):
        epoch = epoch.item() 
    
    confidence_threshold = max(0.6, 0.999 - 0.001 * epoch)
      
    high_confidence_count = 0
    total_count = 0
    
    with torch.no_grad():
        for batch in unlabeled_loader:
            images = batch['image'].float().to(device)
            image_names = batch['image_name']
            
            pred_nerve, pred_cell = model(images)
            
            prob_nerve = torch.sigmoid(pred_nerve)
            prob_cell = torch.sigmoid(pred_cell)
            
            for i, image_name in enumerate(image_names):
                total_count += 1
                
                nerve_prob = prob_nerve[i].cpu().numpy()
                cell_prob = prob_cell[i].cpu().numpy()
                
                nerve_confidence = nerve_prob.mean()
                cell_confidence = cell_prob.mean()
                avg_confidence = (nerve_confidence + cell_confidence) / 2
                
                nerve_binary = (nerve_prob > 0.5).astype(np.uint8) * 255
                cell_binary = (cell_prob > 0.5).astype(np.uint8) * 255
                
                nerve_save_path = os.path.join(args.nerve_pseudo_folder, image_name)
                cell_save_path = os.path.join(args.cell_pseudo_folder, image_name)
                
                cv2.imwrite(nerve_save_path, nerve_binary[0]) 
                cv2.imwrite(cell_save_path, cell_binary[0])   
                
            
                if avg_confidence > confidence_threshold:
                    high_confidence_count += 1
    
    return high_confidence_count

def find_latest_checkpoint(checkpoint_dir):

    if not os.path.exists(checkpoint_dir):
        return None
        
    pattern = os.path.join(checkpoint_dir, "best_checkpoint_*.pkl")
    checkpoint_files = glob.glob(pattern)
    
    if not checkpoint_files:
        return None

    try:
        checkpoint_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        return checkpoint_files[-1]
    except:
        checkpoint_files.sort(key=os.path.getmtime)
        return checkpoint_files[-1]
    
def dilated(label, dilated_pixels):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (dilated_pixels, dilated_pixels))
    dia_label = cv2.dilate(label, kernel)
    # print(dia_label.shape) #(384,384)
    dia_label = dia_label[:,:,np.newaxis]

    return dia_label

import cv2
import numpy as np 
from skimage.morphology import skeletonize
import os

def preprocess(pred_image, mode):
    if mode == "nerve":
        thed_length = 66
    else: 
        thed_length = 40
 
    _, pred_components = cv2.connectedComponents(pred_image)
    num_pred_fibers = np.max(pred_components)
 
    label_lengths = []
    for fiber_label in range(1, num_pred_fibers + 1):
        fiber_mask = np.uint8(pred_components == fiber_label)
        fiber_length = np.sum(fiber_mask)
      
        label_lengths.append(fiber_length)

    pred_new = np.zeros_like(pred_image)   
    for fiber_label in range(1, num_pred_fibers + 1):
       
        fiber_mask = np.uint8(pred_components == fiber_label)
        fiber_length = np.sum(fiber_mask) 
        if fiber_length < thed_length:
            fiber_mask[fiber_mask > 0] = 0
        pred_new += fiber_mask  
          
    pred_new = pred_new * 255
  
    return pred_new  

def After_preprocess(pred_image, mode):
    if mode == "nerve":
        thed_length = 288 # 
        kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        pred_image = cv2.dilate(pred_image, kernel1)  
        pred_image = skeletonize(pred_image).astype(np.uint8) * 255
        pred_image = cv2.dilate(pred_image, kernel2) 
    else:
        thed_length = 40

    _, pred_components = cv2.connectedComponents(pred_image)
    num_pred_fibers = np.max(pred_components)

    label_lengths = []
    for fiber_label in range(1, num_pred_fibers + 1):
        fiber_mask = np.uint8(pred_components == fiber_label)  
        fiber_length = np.sum(fiber_mask)
      
        label_lengths.append(fiber_length)
 
    pred_new = np.zeros_like(pred_image)   
    for fiber_label in range(1, num_pred_fibers + 1):
        fiber_mask = np.uint8(pred_components == fiber_label)
        fiber_length = np.sum(fiber_mask) 
   
        if fiber_length < thed_length:
            fiber_mask[fiber_mask > 0] = 0
        pred_new += fiber_mask  
          
    pred_new = pred_new * 255
  
    return pred_new   
  
def setup_training_resume(args, S_model, optimizer_S):

    start_epoch = 0
    
    if args.resume_training:
        latest_checkpoint = args.S_checkpoint_dir_path_last
        if latest_checkpoint:
            try:
                start_epoch = load_checkpoint_for_resume(
                    latest_checkpoint, S_model, optimizer_S, args
                )
                print(f"Restore: {latest_checkpoint}")
            except Exception as e:
                print(f"Failure: {e}")
                print("Start from scratch...")
                start_epoch = 0
        else:
            print("Find no file")
    else:
        print("Start from scratch...")
    
    return start_epoch

def load_checkpoint_for_resume(checkpoint_path, model, optimizer, args):

    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    args.best_dice = checkpoint['best_dice']
    start_epoch = checkpoint['epoch'] + 1
    
    return start_epoch

def delete_previous_models(folder_path):
 
    files = os.listdir(folder_path)
  
    model_files = [file for file in files if file.endswith('.pkl') or file.endswith('.pth')]

    model_files.sort(key=lambda x: os.path.getmtime(os.path.join(folder_path, x)))
  
    if len(model_files) > 1:
        file_to_delete = model_files[0]
        file_path = os.path.join(folder_path, file_to_delete)
        os.remove(file_path)
        
def save_best_model(args,dice_pred,sen_pred,Train_Down_model):
    if  dice_pred > args.best_dice and sen_pred > args.sen_thed:
        args.best_dice = dice_pred
        args.best_sen = sen_pred
        print("best_dice:{},best_sen:{}".format(args.best_dice,args.best_sen))
        if Train_Down_model == True:
            move_file(args.Down_model_path,args.Down_best_model_path,'lhs_epoch_{}.pkl'.format(args.Down_epoch))
            move_file(args.Down_optimizer_path,args.Down_best_optimizer_path,'lhs_optimizer_{}.pth'.format(args.Down_epoch))
            delete_previous_items(args.Down_best_model_path)
            delete_previous_items(args.Down_best_optimizer_path)
        else:
            move_file(args.Up_model_path,args.Up_best_model_path,'lhs_epoch_{}.pkl'.format(args.Up_epoch))
            move_file(args.Up_optimizer_path,args.Up_best_optimizer_path,'lhs_optimizer_{}.pth'.format(args.Up_epoch))
            delete_previous_items(args.Up_best_model_path)
            delete_previous_items(args.Up_best_optimizer_path)

def delete_previous_items(folder_path):

    files = os.listdir(folder_path)

    model_files = [file for file in files if file.endswith('.pkl') or file.endswith('.pth')]

    model_files.sort(key=lambda x: os.path.getmtime(os.path.join(folder_path, x)))

    if len(model_files) > 5:
        file_to_delete = model_files[0]
        file_path = os.path.join(folder_path, file_to_delete)
        os.remove(file_path)
 
def move_file(source_folder, target_folder, file_name):
    print("Perform move file!")
    source_path = os.path.join(source_folder,file_name)
    target_path = os.path.join(target_folder,file_name)
    shutil.copy(source_path, target_path)

def clear_folder(folder_path):

    files = os.listdir(folder_path)
    
    for file in files:
        file_path = os.path.join(folder_path, file)
        os.remove(file_path)

def rename(folder_path):

    files = os.listdir(folder_path)
    

    for file in files:
        os.rename(file,"1_"+file)


def resize(image,label,img_size):

    resized_image = cv2.resize(image, img_size)
    resized_label = cv2.resize(label, img_size)
    _, resized_label = cv2.threshold(resized_label, 128, 255, cv2.THRESH_BINARY) 

    return resized_image,resized_label


def center_crop(image, label, new_h, new_w):
    h, w, _ = image.shape

    top = (h - new_h) // 2
    left = (w - new_w) // 2
    bottom = top + new_h
    right = left + new_w

    cropped_image = image[top:bottom, left:right, :]
    cropped_label = label[top:bottom, left:right, :]

    return cropped_image, cropped_label

def dialated_plain(image,label,dialate_pixels):

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (dialate_pixels, dialate_pixels))

    dia_label = cv2.dilate(label, kernel)
    dia_label = dia_label // 255
    dia_label = dia_label[:,:,np.newaxis] #(H,W,1)

    dia_image = (image * dia_label).astype(np.uint8)
    
    return dia_image

def compute_sdf(img_gt, out_shape):
    """
    compute the signed distance map of binary mask
    input: segmentation, shape = (batch_size, x, y, z)
    output: the Signed Distance Map (SDM) 
    sdf(x) = 0; x in segmentation boundary
             -inf|x-y|; x in segmentation
             +inf|x-y|; x out of segmentation
    normalize sdf to [-1,1]

    """

    img_gt = img_gt.astype(np.uint8)
    normalized_sdf = np.zeros(out_shape)

    for b in range(out_shape[0]): # batch size
        for c in range(out_shape[1]):
            posmask = img_gt[b].astype(bool)
            if posmask.any():
                negmask = ~posmask
                posdis = distance(posmask)
                negdis = distance(negmask)
                boundary = skimage_seg.find_boundaries(posmask, mode='inner').astype(np.uint8)
                sdf = (negdis-np.min(negdis))/(np.max(negdis)-np.min(negdis)) - (posdis-np.min(posdis))/(np.max(posdis)-np.min(posdis))
                sdf[boundary==1] = 0
                sdf[sdf < 0] = 0 
                # print(np.unique(sdf))
                normalized_sdf[b][c] = sdf
                # assert np.min(sdf) == -1.0, print(np.min(posdis), np.max(posdis), np.min(negdis), np.max(negdis))
                assert np.max(sdf) ==  1.0, print(np.min(posdis), np.min(negdis), np.max(posdis), np.max(negdis))

    return normalized_sdf

def get_SDF_data(image,label,beta):
    image = np.transpose(image, (2,0,1)) #[C,H,W]
    image = image[np.newaxis,:,:,:] # [1,C,H,W]
    label = label.squeeze()
    label = label[np.newaxis,np.newaxis,:,:] # [1,1,H,W]
    SDF_label = compute_sdf(label, label.shape)    

    SDF_label = 1 / (1 + np.log(1 + beta * np.abs(SDF_label)))
    SDF_image = (image * SDF_label).astype(np.uint8).squeeze()
    SDF_label = SDF_label.squeeze() * 255
    SDF_label = SDF_label.astype(np.uint8)
    if len(SDF_image.shape) == 3:
        SDF_image = np.transpose(SDF_image, (1,2,0)) #[C,H,W] -> [H,W,C] 

    return SDF_image,SDF_label

def min_max_normalize(tensor):
    return (tensor - tensor.min()) / (tensor.max() - tensor.min())

def center_crop_and_pad2(image, label,th, tw):
 
    h, w = image.shape[:2]

    x1 = max(0, int((w - tw) / 2))
    y1 = max(0, int((h - th) / 2))
    

    x2 = min(w, x1 + tw)
    y2 = min(h, y1 + th)
    
    if len(image.shape) == 2:
        cropped_image = np.zeros((th, tw), dtype=image.dtype)
    else:

        cropped_image = np.zeros((th, tw, image.shape[2]), dtype=image.dtype)
    cropped_label =  np.zeros((th, tw),dtype=label.dtype)

    cropped_image[(th - (y2 - y1)) // 2:(th - (y2 - y1)) // 2 + (y2 - y1),
                  (tw - (x2 - x1)) // 2:(tw - (x2 - x1)) // 2 + (x2 - x1)] = image[y1:y2, x1:x2]
    cropped_label[(th - (y2 - y1)) // 2:(th - (y2 - y1)) // 2 + (y2 - y1),
                  (tw - (x2 - x1)) // 2:(tw - (x2 - x1)) // 2 + (x2 - x1)] = label[y1:y2, x1:x2]
    
    return cropped_image,cropped_label

def center_crop_and_pad3(image, label,random_label,th, tw):

    h, w = image.shape[:2]

    x1 = max(0, int((w - tw) / 2))
    y1 = max(0, int((h - th) / 2))

    x2 = min(w, x1 + tw)
    y2 = min(h, y1 + th)
    
    random_label = np.resize(random_label, (h, w))
    
    if len(image.shape) == 2:
        cropped_image = np.zeros((th, tw), dtype=image.dtype)
    else:
  
        cropped_image = np.zeros((th, tw, image.shape[2]), dtype=image.dtype)
        
    cropped_label =  np.zeros((th, tw), dtype=label.dtype)
    cropped_random_label = np.zeros((th, tw), dtype=random_label.dtype)  

    cropped_image[(th - (y2 - y1)) // 2:(th - (y2 - y1)) // 2 + (y2 - y1),
                  (tw - (x2 - x1)) // 2:(tw - (x2 - x1)) // 2 + (x2 - x1)] = image[y1:y2, x1:x2]
    cropped_label[(th - (y2 - y1)) // 2:(th - (y2 - y1)) // 2 + (y2 - y1),
                  (tw - (x2 - x1)) // 2:(tw - (x2 - x1)) // 2 + (x2 - x1)] = label[y1:y2, x1:x2]
    cropped_random_label[(th - (y2 - y1)) // 2:(th - (y2 - y1)) // 2 + (y2 - y1),
                  (tw - (x2 - x1)) // 2:(tw - (x2 - x1)) // 2 + (x2 - x1)] = random_label[y1:y2, x1:x2]
    
    return cropped_image,cropped_label,cropped_random_label


def calculate_mean_and_std(metric):
    """
    input: type--list; 
    content--[[A,B,C,D],[A,B,C,D],[A,B,C,D]] 


    mean_metric : For single target(1), we compute sen.
    mean_total : For All targets(C), we compute sen. 
    if there is only one target, they are the same.
    """
    # 只有tensor或者array可axis操作或者shape
    metric = np.array(metric)
    # print(metric.shape)
    # print("metric.shape:{}".format(metric.shape))
    # Per class metric of mean±std.
    mean_metric = np.mean(metric,axis=0)
    # print(mean_metric)
    std_metric = np.std(metric,axis=0)
    # Total metric of mean±std.
    mean_total = np.mean(mean_metric,axis=0)
    std_total = np.std(std_metric,axis=0)

    return mean_total,std_total,mean_metric, std_metric

import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import binary_fill_holes
import random

def create_nonzero_mask(data):
  
    assert len(data.shape) == 3, "data must have shape (H, W, C)"
    nonzero_mask = np.zeros(data.shape[:2], dtype=bool)
    for c in range(data.shape[-1]):
        this_mask = data[..., c] != 0
        nonzero_mask = nonzero_mask | this_mask
    nonzero_mask = binary_fill_holes(nonzero_mask)
    return nonzero_mask

def target_get_bbox_center(roi_size,image,nonzero_mask):
    H, W = image.shape[0],image.shape[1]
  
    true_coords = np.where(nonzero_mask)
  
    if true_coords[0].size > 0:

        random_index = np.random.choice(true_coords[0].size)
     
        x_coord = true_coords[1][random_index]
        y_coord = true_coords[0][random_index]
    else:
        if image.shape[0] < roi_size[0] or image.shape[1] < roi_size[1]:
            x_coord = W // 2
            y_coord = H // 2
        else:

            x_coord = random.randint(roi_size[1] // 2, W - roi_size[1] // 2)
            y_coord = random.randint(roi_size[0] // 2, H - roi_size[0] // 2)

    return x_coord,y_coord

def target_get_cropping_bbox(image,label,crop_size):
    """
    Generate the cropping bounding box based on the center coordinate and crop size.
    
    :param image_shape: A tuple (H, W) representing the shape of the image.
    :param crop_size: A tuple (crop_height, crop_width) representing the size of the crop.
    :param center_coord: A tuple (x, y) representing the center coordinate of the crop.
    
    :return: A list of coordinates [[min_z_idx, max_z_idx], [min_x_idx, max_x_idx], [min_y_idx, max_y_idx]]
    """
    H, W = image.shape[0],image.shape[1]
    crop_height, crop_width = crop_size
    nonzero_mask = create_nonzero_mask(label)
    x, y = target_get_bbox_center(crop_size,image,nonzero_mask)
    
    # Calculate the top-left corner of the crop
    half_crop_height = crop_height // 2
    half_crop_width = crop_width // 2
    
    # Calculate the minimum and maximum indices for cropping
    min_y_idx = max(y - half_crop_height, 0)
    max_y_idx = min(y + half_crop_height, H)
    
    min_x_idx = max(x - half_crop_width, 0)
    max_x_idx = min(x + half_crop_width, W)
    
    # Generate the bounding box coordinates
    bbox = [[min_y_idx, max_y_idx], [min_x_idx, max_x_idx]]
    
    return bbox

def random_get_cropping_bbox(image,crop_size):
    """
    Generate the cropping bounding box based on the center coordinate and crop size.
    
    :param image_shape: A tuple (H, W) representing the shape of the image.
    :param crop_size: A tuple (crop_height, crop_width) representing the size of the crop.
    :param center_coord: A tuple (x, y) representing the center coordinate of the crop.
    
    :return: A list of coordinates [[min_z_idx, max_z_idx], [min_x_idx, max_x_idx], [min_y_idx, max_y_idx]]
    """
    H, W = image.shape[0],image.shape[1]
    crop_height, crop_width = crop_size

    half_crop_height = crop_height // 2 
    half_crop_width = crop_width // 2


    if image.shape[0] < crop_height or image.shape[1] < crop_width:
        x = W // 2
        y = H // 2
    else:
        x = random.randint(half_crop_width, W - half_crop_width)
        y = random.randint(half_crop_height, H - half_crop_height)
    
    # Calculate the minimum and maximum indices for cropping
    min_y_idx = max(y - half_crop_height, 0)
    max_y_idx = min(y + half_crop_height, H)
    
    min_x_idx = max(x - half_crop_width, 0)
    max_x_idx = min(x + half_crop_width, W)
    
    # Generate the bounding box coordinates
    bbox = [[min_y_idx, max_y_idx], [min_x_idx, max_x_idx]]
    
    return bbox

def crop_to_bbox(image, bbox):

    return image[bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1]]

def zero_pad_image(image, roi_size):

    if image.shape[0] != roi_size[0] or image.shape[1] != roi_size[1]:

        original_height, original_width = image.shape[:2]
        target_height, target_width = roi_size


        pad_height = max(target_height - original_height, 0)
        pad_width = max(target_width - original_width, 0)

     
        top_pad = pad_height // 2
        bottom_pad = pad_height - top_pad
        left_pad = pad_width // 2
        right_pad = pad_width - left_pad

      
        if len(image.shape) == 3:  
            image = np.pad(image, ((top_pad, bottom_pad), (left_pad, right_pad), (0, 0)), mode='constant', constant_values=0)
        elif len(image.shape) == 2:  
            image = np.pad(image, ((top_pad, bottom_pad), (left_pad, right_pad)), mode='constant', constant_values=0)

    return image

def crop_images_and_label(image, label, image_dia, crop_size):

    if np.random.rand() < 0.3:

        bbox = target_get_cropping_bbox(image,label,crop_size)
        cropped_image = crop_to_bbox(image, bbox)
        cropped_image = zero_pad_image(cropped_image, crop_size)
        cropped_label = crop_to_bbox(label, bbox)
        cropped_label = zero_pad_image(cropped_label, crop_size)
        cropped_image_dia = crop_to_bbox(image_dia, bbox)
        cropped_image_dia = zero_pad_image(cropped_image_dia, crop_size)

    else:

        bbox = random_get_cropping_bbox(image,crop_size)
        cropped_image = crop_to_bbox(image, bbox)
        cropped_image = zero_pad_image(cropped_image, crop_size)
        cropped_label = crop_to_bbox(label, bbox)
        cropped_label = zero_pad_image(cropped_label, crop_size)
        cropped_image_dia = crop_to_bbox(image_dia, bbox)
        cropped_image_dia = zero_pad_image(cropped_image_dia, crop_size)

    return cropped_image, cropped_label, cropped_image_dia

