import cv2
import os

def img2heatmap(img_path, mask_path):
    img_path = os.path.join(os.getcwd(),img_path)
    mask_path = os.path.join(os.getcwd(),mask_path)
    print("img_path", img_path)
    print("mask_path", mask_path)
    img = cv2.imread(str(img_path), 0)
    img = cv2.resize(img, (208, 176), interpolation = cv2.INTER_CUBIC)
    heatmap=  cv2.imread(str(mask_path), 0)
    heatmap = cv2.resize(heatmap, (208, 176), interpolation = cv2.INTER_CUBIC)
    img = cv2.applyColorMap(img, cv2.COLORMAP_BONE)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_PINK)
    image = 0.6*img+0.4*heatmap
    return image