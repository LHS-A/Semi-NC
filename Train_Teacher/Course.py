import os
import shutil
import numpy as np
import cv2
from scipy.ndimage import distance_transform_edt as distance
from skimage import segmentation as skimage_seg
import random
import matplotlib.pyplot as plt

 
def compute_sdf(img_gt, out_shape):
    img_gt = img_gt.astype(np.uint8)
    normalized_sdf = np.zeros(out_shape)

    for b in range(out_shape[0]):
        for c in range(out_shape[1]):
            posmask = img_gt[b].astype(bool)
            if posmask.any():
                negmask = ~posmask
                posdis = distance(posmask)
                negdis = distance(negmask)
                boundary = skimage_seg.find_boundaries(posmask, mode='inner').astype(np.uint8)
                sdf = (negdis - np.min(negdis)) / (np.max(negdis) - np.min(negdis)) - \
                      (posdis - np.min(posdis)) / (np.max(posdis) - np.min(posdis))
                sdf[boundary == 1] = 0
                sdf[sdf < 0] = 0
                normalized_sdf[b][c] = sdf
                assert np.max(sdf) == 1.0, print(np.min(posdis), np.min(negdis), np.max(posdis), np.max(negdis))

    return normalized_sdf

def get_SDF_data(image, label, beta):
    if len(image.shape) == 3:
        image = np.transpose(image, (2, 0, 1))
        image = image[np.newaxis, :, :, :]
    else:
        image = image[np.newaxis, np.newaxis, :, :]
    label = label.squeeze() 
    label = label[np.newaxis, np.newaxis, :, :]
    SDF_label = compute_sdf(label, label.shape)
    # SDF_label = np.exp(-beta * SDF_label)   
    SDF_label = 1 / (1 + np.log(1 + beta * np.abs(SDF_label))) # Empirically, beta is in [1-6]; set to 3 for inference
    SDF_image = (image * SDF_label).astype(np.uint8).squeeze()
    SDF_label = SDF_label.squeeze() * 255
    SDF_label = SDF_label.astype(np.uint8)
    if len(SDF_image.shape) == 3:
        SDF_image = np.transpose(SDF_image, (1, 2, 0))
    return SDF_image, SDF_label 
   
# ========================== Func 1: original label (already saved as label) ==========================
# Called in main; result_1_label.png is saved

# ========================== Func 2: rotated label (random angle [-30, 30]) ==========================
def process_rotated_label(image, label, beta):
    angle = random.uniform(-30, 30)  # ✅ random angle: -30 ~ 30
    h, w = label.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_label = cv2.warpAffine(label, M, (w, h), borderValue=0)
    SDF_image, SDF_label = get_SDF_data(image, rotated_label, beta)
    # output_path = r"C:\Users\lenovo\Desktop\hongshuo\result_rotated.png"
    # cv2.imwrite(output_path, res_sdf_label) 
    # label_output_path = r"C:\Users\lenovo\Desktop\hongshuo\result_rotated_label.png"
    # cv2.imwrite(label_output_path, rotated_label) 
    # print(f"✅ Rotation angle：{angle:.2f}°, SDF result saved to：{output_path}")
    # print(f"✅ Rotated label saved to：{label_output_path}")
    return SDF_image, SDF_label, rotated_label

# ========================== Func 3: translated label (shift [-100, 100]) ==========================
def process_translated_label(image, label, beta):
    tx = random.randint(-100, 100)  # ✅ random tx
    ty = random.randint(-100, 100)  # ✅ random ty
    h, w = label.shape
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    translated_label = cv2.warpAffine(label, M, (w, h), borderValue=0)
    SDF_image, SDF_label = get_SDF_data(image, translated_label, beta)
    # output_path = r"C:\Users\lenovo\Desktop\hongshuo\result_translated.png"
    # cv2.imwrite(output_path, res_sdf_label) 
    # label_output_path = r"C:\Users\lenovo\Desktop\hongshuo\result_translated_label.png"
    # cv2.imwrite(label_output_path, translated_label)
    # print(f"✅ Shift (tx={tx}, ty={ty}), SDF result saved to：{output_path}")
    # print(f"✅ Translated label saved to：{label_output_path}")
    return SDF_image, SDF_label, translated_label

# ========================== Func 4: random interference + 3 random patches (20~70) ===========================
def process_random_interference_patch(image, original_label, beta, interference_folder):
    try:
        label_files = [os.path.join(interference_folder, f) for f in os.listdir(interference_folder)
                       if f.endswith(('.png', '.jpg', '.bmp')) and os.path.isfile(os.path.join(interference_folder, f))]
    except Exception as e:
        # print(f"❌ Cannot read interference label folder：{interference_folder}, error：{e}")
        return
    if not label_files:
        # print(f"❌ Interference label folder empty：{interference_folder}")
        return
 
    selected_interference_path = random.choice(label_files)
    # print(f"🔹 Random interference label file：{selected_interference_path}")
    interference_label = cv2.imread(selected_interference_path, 0)
    if interference_label is None:
        # print(f"❌ Cannot read interference label：{selected_interference_path}")
        return
    _, interference_label = cv2.threshold(interference_label, 127, 255, cv2.THRESH_BINARY)

    h, w = interference_label.shape
    patches = []
    masked_interference = np.zeros_like(interference_label)

    for _ in range(3):
        # Random patch size: square, side 20~70
        patch_size = random.randint(20, 70)

        # Random top-left x, y inside image
        x = random.randint(0, max(0, w - patch_size))
        y = random.randint(0, max(0, h - patch_size))

        # Crop patch from interference label
        patch = interference_label[y:y+patch_size, x:x+patch_size]
        # Place patch into masked_interference
        masked_interference[y:y+patch_size, x:x+patch_size] = patch

    # Overlay patches onto original label (OR operation, keep original foreground)
    new_label = cv2.bitwise_or(original_label, masked_interference)
    _, new_label = cv2.threshold(new_label, 127, 255, cv2.THRESH_BINARY)
 
    # Compute SDF
    SDF_image, SDF_label = get_SDF_data(image, new_label, beta)
 
    # Save SDF result
    # output_sdf_path = r"C:\Users\lenovo\Desktop\hongshuo\result_random_interference_patch.png"
    # cv2.imwrite(output_sdf_path, res_sdf_label)
    # print(f"✅ Synthetic interference label SDF result saved to：{output_sdf_path}")

    # Save synthetic label (original + random patches)
    # output_label_path = r"C:\Users\lenovo\Desktop\hongshuo\result_random_interference_patch_label.png"
    # cv2.imwrite(output_label_path, new_label)
    # print(f"✅ Synthetic interference label saved to：{output_label_path}")
    return SDF_image, SDF_label, new_label 

# ========================== Func 5: random truncation (4~8 squares, side 20~70) ==========================
def process_label_with_truncations(image, label, beta): 
    label_copy = label.copy()
    h, w = label_copy.shape
    foreground_points = np.argwhere(label_copy == 255)
    num_truncations = random.randint(4, 8)  # ✅ random 4~8 truncations
    truncated_label = label_copy.copy() 
    rect_size = random.randint(20, 70)  # ✅ each square side 20~70

    for _ in range(num_truncations):
        if len(foreground_points) == 0:
            break
        idx = random.randint(0, len(foreground_points) - 1)
        pt = foreground_points[idx]
        foreground_points = np.delete(foreground_points, idx, axis=0)
        y, x = pt

        half = rect_size // 2
        x_min = max(0, x - half)
        x_max = min(w, x + half)
        y_min = max(0, y - half)
        y_max = min(h, y + half)

        truncated_label[y_min:y_max, x_min:x_max] = 0

    SDF_image, SDF_label = get_SDF_data(image, truncated_label, beta)
    # output_sdf_path = r"C:\Users\lenovo\Desktop\hongshuo\result_truncated_3cuts.png"
    # cv2.imwrite(output_sdf_path, res_sdf_label)
    # output_label_path = r"C:\Users\lenovo\Desktop\hongshuo\result_truncated_3cuts_label.png"
    # cv2.imwrite(output_label_path, truncated_label)
    # print(f"✅ Truncated label SDF result saved to：{output_sdf_path}")
    # print(f"✅ Truncated label saved to：{output_label_path}")
    return SDF_image, SDF_label, truncated_label 

def show_pair(img1, img2, title1='Image 1', title2='Image 2'):
    """Display two images; function returns only after windows are closed"""
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img1, cmap='gray')
    plt.title(title1)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(img2, cmap='gray')
    plt.title(title2)
    plt.axis('off')
    plt.tight_layout()
    plt.show()          # Block until window is closed
    print("Window closed, continuing...")

def Course_Augmentation(image, label, beta, interference_folder):
    """ 
    Curriculum augmentation controller
    :param image: original image, uint8 ndarray
    :param label: original label, uint8 ndarray
    :param beta:  SDF parameter 
    """ 
    # === Func 1: original label ===
    # print("🔹 Processing original label...") 
    Ori_SDF_image, Ori_SDF_label = get_SDF_data(image, label, beta)  
    # show_pair(Ori_SDF_image, Ori_SDF_label)
    
    # === Func 2: rotated label (random angle) === 
    # print("🔹 Processing rotated label...")
    SDF_image_rotate, SDF_label_rotate, label_rotate = process_rotated_label(image, label, beta)
    # show_pair(SDF_image_rotate, SDF_label_rotate)

    # === Func 3: translated label (random tx, ty) ===
    # print("🔹 Processing translated label...")
    SDF_image_trans, SDF_label_trans, label_trans = process_translated_label(image, label_rotate, beta)
    # show_pair(SDF_image_trans, SDF_label_trans)
 
    # === Func 4: random interference label + 3 patches (random squares 20~70 px) ===
    # print("🔹 Processing random interference label + 3 patches...") 
    SDF_image_interference, SDF_label_interference, label_interference = process_random_interference_patch(image, label_trans, beta, interference_folder)
    # show_pair(SDF_image_interference, SDF_label_interference)
    
    # === Func 5: random truncation (4~8 squares, 20~70 px) ===  
    # print("🔹 Processing Func 5: random truncation of label...")  
    SDF_image, SDF_label, truncated_label = process_label_with_truncations(image, label_interference, beta)       
    # show_pair(SDF_image, SDF_label) 
         
    return SDF_image
 
# ----------- Local test entry -----------   
if __name__ == "__main__": 
    image = cv2.imread(r"/data/Desktop/Semi-NC/Dataset/CORN/train/image/BHR(99).png", 0)
    CNs_label = cv2.imread(r"/data/Desktop/Semi-NC/Dataset/CORN/train/nerve_label/BHR(99).png", 0)
    LCs_label = cv2.imread(r"/data/Desktop/Semi-NC/Dataset/CORN/train/cell_label/BHR(99).png", 0)
    label = cv2.add(CNs_label, LCs_label)  
    beta = 3
    SDF_image = Course_Augmentation(image, label, beta)