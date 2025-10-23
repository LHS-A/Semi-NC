# -- coding: utf-8 --
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import albumentations as A
from PIL import Image, ImageEnhance
import torch
import torchvision.transforms as transforms

aug = transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (12,12))

def flip_3(image,label1,label2, flip_code = random.choice([0,1,-1]),p = random.random()):
    if p > 0.5:                      
        flipped_image =cv2.flip(image, flip_code)
        flipped_label1 =cv2.flip(label1, flip_code)
        flipped_label2 =cv2.flip(label2, flip_code)
    else:
        flipped_image = image
        flipped_label1 = label1
        flipped_label2 = label2

    return flipped_image,flipped_label1,flipped_label2

def rotate_3(image: np.ndarray,
             nerve_label: np.ndarray,
             cell_label: np.ndarray,
             angle: int = None,
             p: float = None):

    if angle is None:
        angle = random.randint(-90, 90)
    if p is None:
        p = random.random()

    if p > 0.5:                       # 50% 概率做旋转
        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        rotated_image = cv2.warpAffine(image, M, (w, h),
                                       flags=cv2.INTER_LINEAR,
                                       borderMode=cv2.BORDER_REFLECT)
        rotated_nerve = cv2.warpAffine(nerve_label, M, (w, h),
                                       flags=cv2.INTER_NEAREST,
                                       borderMode=cv2.BORDER_REFLECT)
        rotated_cell = cv2.warpAffine(cell_label, M, (w, h),
                                      flags=cv2.INTER_NEAREST,
                                      borderMode=cv2.BORDER_REFLECT)
    else:
        rotated_image, rotated_nerve, rotated_cell = image, nerve_label, cell_label

    return rotated_image, rotated_nerve, rotated_cell

def apply_augmentations_KD(image,nerve_label,cell_label):
    transform = A.Compose([
        A.ColorJitter(always_apply=False, p=0.5, brightness=(0.8, 1.2), contrast=(0.8, 1.2), saturation=(0.8, 1.2), hue=(-0.2, 0.2)),
        A.GaussNoise(always_apply=False, p=0.5, var_limit=(10.0, 50.0), per_channel=True, mean=0.0),
        A.GaussianBlur(always_apply=False, p=0.5, blur_limit=(3, 7), sigma_limit=(0.0, 0)),
        A.RandomBrightnessContrast(always_apply=False, p=0.5, brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), brightness_by_max=True),
    ])
   
    rot_img, rot_nerve, rot_cell = rotate_3(image,nerve_label,cell_label)
   
    image,nerve_label,cell_label = flip_3(rot_img, rot_nerve, rot_cell)
    # image = image[:,:,np.newaxis]
    # image = np.repeat(image,3,axis=-1) #（H,W,3）
    augmented_image = transform(image=image)['image']

    return augmented_image,nerve_label,cell_label

def randomHueSaturationValue(image, hue_shift_limit=(-180, 180),
                             sat_shift_limit=(-255, 255),
                             val_shift_limit=(-255, 255), u=0.5):
    if np.random.random() < u:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(image)
        hue_shift = np.random.randint(hue_shift_limit[0], hue_shift_limit[1] + 1)
        hue_shift = np.uint8(hue_shift)
        h += hue_shift
        sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
        s = cv2.add(s, sat_shift)
        val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
        v = cv2.add(v, val_shift)
        image = cv2.merge((h, s, v))
        # image = cv2.merge((s, v))
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    return image


def randomShiftScaleRotate(image, mask,
                           shift_limit=(-0.0, 0.0),
                           scale_limit=(-0.0, 0.0),
                           rotate_limit=(-0.0, 0.0),
                           aspect_limit=(-0.0, 0.0),
                           borderMode=cv2.BORDER_CONSTANT, u=0.5):
    if np.random.random() < u:
        height, width, channel = image.shape

        angle = np.random.uniform(rotate_limit[0], rotate_limit[1])
        scale = np.random.uniform(1 + scale_limit[0], 1 + scale_limit[1])
        aspect = np.random.uniform(1 + aspect_limit[0], 1 + aspect_limit[1])
        sx = scale * aspect / (aspect ** 0.5)
        sy = scale / (aspect ** 0.5)
        dx = round(np.random.uniform(shift_limit[0], shift_limit[1]) * width)
        dy = round(np.random.uniform(shift_limit[0], shift_limit[1]) * height)

        cc = np.math.cos(angle / 180 * np.math.pi) * sx
        ss = np.math.sin(angle / 180 * np.math.pi) * sy
        rotate_matrix = np.array([[cc, -ss], [ss, cc]])

        box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
        box1 = box0 - np.array([width / 2, height / 2])
        box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])

        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0, box1)
        image = cv2.warpPerspective(image, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                    borderValue=(
                                        0, 0,
                                        0,))
        mask = cv2.warpPerspective(mask, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                   borderValue=(
                                       0, 0,
                                       0,))

    return image, mask

