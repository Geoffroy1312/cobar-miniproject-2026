import cv2
import numpy as np

def detect_dragonfly(frame, surface_min=50, surface_max=10000):
    """
    Detect the dragon fly head
    """
    
    if frame.dtype != np.uint8:
        if frame.max() <= 1.0:
            frame = (frame * 255).astype(np.uint8)
        else:
            frame = frame.astype(np.uint8)

    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    
    #define lower and upper bounds for red
    lower_red_1 = np.array([0, 100, 50])
    upper_red_1 = np.array([10, 255, 255])
    
    lower_red_2 = np.array([170, 100, 50])
    upper_red_2 = np.array([180, 255, 255])
    
    #isolate red channel
    mask1 = cv2.inRange(hsv, lower_red_1, upper_red_1)
    mask2 = cv2.inRange(hsv, lower_red_2, upper_red_2)
    
    #combine both masks
    mask = cv2.bitwise_or(mask1, mask2)
    
    #remove small red eyes
    kernel = np.ones((15, 15), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    #find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    contours_valid = []
    
    #avoid taking small contours or contours too large
    for contour in contours:
        surface = cv2.contourArea(contour)
        if surface_min < surface < surface_max:
            contours_valid.append(contour)
            
    #boolean to check if the dragonfly is present, contours valid is only used to debug
    presence = len(contours_valid) > 0
    
    return presence, contours_valid

