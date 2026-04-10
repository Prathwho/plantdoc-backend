import numpy as np
import cv2

def is_plant_heuristic(image_path):
    img = cv2.imread(image_path)
    if img is None: return False
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Green, yellow, brown color ranges
    # Hue usually 20 to 90 out of 180
    lower = np.array([20, 20, 20])
    upper = np.array([90, 255, 255])
    
    mask = cv2.inRange(hsv, lower, upper)
    ratio = cv2.countNonZero(mask) / (img.shape[0] * img.shape[1])
    
    return ratio

print(f"Ratio of plant colors: {is_plant_heuristic('test.jpg')}")
