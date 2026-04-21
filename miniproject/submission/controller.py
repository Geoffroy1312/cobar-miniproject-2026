import numpy as np
from miniproject.simulation import MiniprojectSimulation
import cv2
import matplotlib.pyplot as plt

def detect_triangles(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
    # 2. Extraire uniquement le canal V (Index 2)
    # Cela donne une image en niveaux de gris basée uniquement sur la luminosité
    v_channel = hsv_image[:, :, 2].copy() 
    
    # 3. Identifier les pixels verts
    # On définit la plage du vert. On garde une saturation basse (ex: 40) 
    # pour s'assurer de bien attraper les verts même un peu délavés.
    lower_green = np.array([35, 40, 0])
    upper_green = np.array([85, 255, 255])
    
    # green_mask vaut 255 (blanc) là où c'est vert, et 0 (noir) ailleurs
    green_mask = cv2.inRange(hsv_image, lower_green, upper_green)
    
    # 4. Soustraire le vert du canal V
    # Partout où le masque vert est activé (supérieur à 0), on force le pixel du canal V à être noir (0)
    v_channel[green_mask <= 0] = 0
    #augmenter la luminosité du canal V pour mieux voir les détails
    v_channel = cv2.equalizeHist(v_channel)
    v_channel[v_channel <= 250] = 0
    #remove small blobs
    #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    #v_channel = cv2.morphologyEx(v_channel, cv2.MORPH_OPEN, kernel)
    #v_channel = cv2.morphologyEx(v_channel, cv2.MORPH_CLOSE, kernel)

    #detecte les triangles 
    contours, _ = cv2.findContours(v_channel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    big_contours = []
    for contour in contours:
        epsilon = 0.04 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) >= 3 and cv2.contourArea(approx) > 20:  # Si le contour a 3 sommets, c'est un triangle
            
            #cv2.drawContours(image, [approx], 0, (255, 0, 0), 2)  # Dessine le triangle en rouge
            big_contours.append(approx)
    
    return big_contours


def odor_intensity_to_control_signal(
    odor_intensities,
    attractive_gain=-500,
    aversive_gain=80,
):
    """Convert odor sensor readings to a turning control signal."""
    
    # Always process the first odor dimension (attractive)
    attractive_intensities = np.average(
        odor_intensities[:, 0].reshape(2, 2), axis=0, weights=[9, 1]
    )
    attractive_bias = (
        attractive_gain
        * (attractive_intensities[0] - attractive_intensities[1])
        / attractive_intensities.mean()
        if attractive_intensities.mean() != 0
        else 0
    )

    # Check if a second odor dimension (aversive) exists
    if odor_intensities.shape[1] > 1:
        aversive_intensities = np.average(
            odor_intensities[:, 1].reshape(2, 2), axis=0, weights=[10, 0]
        )
        aversive_bias = (
            aversive_gain
            * (aversive_intensities[0] - aversive_intensities[1])
            / aversive_intensities.mean()
            if aversive_intensities.mean() != 0
            else 0
        )
    else:
        aversive_bias = 0

    effective_bias = aversive_bias + attractive_bias
    effective_bias_norm = np.tanh(effective_bias**2) * np.sign(effective_bias)
    assert np.sign(effective_bias_norm) == np.sign(effective_bias)

    control_signal = np.ones(2)
    side_to_modulate = int(effective_bias_norm > 0)
    modulation_amount = np.abs(effective_bias_norm) * 0.8
    control_signal[side_to_modulate] -= modulation_amount
    
    return control_signal


class Controller:
    def __init__(self, sim: MiniprojectSimulation):
        # you may also implement your own turning controller
        from flygym.examples.locomotion import TurningController

        self.turning_controller = TurningController(sim.timestep)

    def step(self, sim: MiniprojectSimulation):
        # implement your control algorithm here
        olfaction = sim.get_olfaction(sim.fly.name)
        vision = sim.get_raw_vision(sim.fly.name)
        vision_left = vision[0]
        vision_right = vision[1]
        contours_left = detect_triangles(vision_left)
        contours_right = detect_triangles(vision_right)
        #check if a triangle bigger than a threshold is detected in the left eye, then turn left, if in the right eye, turn right
        if len(contours_left) > 0 and len(contours_right) == 0:
            olfaction[:, 0] *= 1.5
            
        elif len(contours_right) > 0 and len(contours_left) == 0:
            olfaction[:, 0] *= 0.5
            
        elif len(contours_left) > 0 and len(contours_right) > 0:
            olfaction[:, 0] *= 1.2
            
                
        else:
            olfaction[:, 0] *= 1
            
      

        
        drives = odor_intensity_to_control_signal(olfaction)
        
        joint_angles, adhesion = self.turning_controller.step(drives)
        return joint_angles, adhesion
