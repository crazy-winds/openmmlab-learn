import os
import glob
import cv2 as cv
import numpy as np
from PIL import Image


def processing(root_data, img_suffix=".bmp", target_data="./pre_mask"):
    """ 预处理mask

    Args:
        root(str): 原mask路径
        target_data(str): 目标mask路径
    """
    imgs = glob.glob(os.path.join(root_data, f"*{img_suffix}"))
    if not os.path.exists(target_data):
        os.mkdir(target_data)
    
    for img_path in imgs:
        # (H, W, C)
        img = cv.imread(img_path)
        mask = np.dot((img > 100), np.array([[4], [2], [1]])).astype(np.uint8)
        if (mask.shape[:2] != img.shape[:2]):
            print(img_path)
            
        cv.imwrite(os.path.join(target_data, img_path.split("/")[-1].split(".")[0] + ".png"), mask)

processing("masks")

imgs = glob.glob(os.path.join("images", "*.jpg"))
for img_path in imgs:
    # (H, W, C)
    mask_path = os.path.join("pre_mask", img_path.split("/")[-1].split(".")[0] + ".png")
    img, mask = (
        cv.imread(img_path).shape,
        cv.imread(mask_path).shape
    )
    if (img != mask):
        print(img, mask)
        img = cv.imread(img_path)
        mask = np.dot((img > 100), np.array([[4], [2], [1]])).astype(np.uint8)

        cv.imwrite(os.path.join("pre_mask", img_path.split("/")[-1].split(".")[0] + ".png"), mask)
print("over")

imgs = glob.glob(os.path.join("images", "*.jpg"))
for img_path in imgs:
    # (H, W, C)
    mask_path = os.path.join("pre_mask", img_path.split("/")[-1].split(".")[0] + ".png")
    img, mask = (
        cv.imread(img_path).shape,
        cv.imread(mask_path).shape
    )
    if (img != mask):
        print(img, mask)