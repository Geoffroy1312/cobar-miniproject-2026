import numpy as np
from miniproject.simulation import MiniprojectSimulation
import cv2
import matplotlib.pyplot as plt


def detect_triangles_combined(image):
    """
    Détecte les triangles sur une image concaténée (gauche + droite).
    La distance retournée est négative pour la gauche, positive pour la droite.
    """
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
    v_channel = hsv_image[:, :, 1].copy() 
    
    lower_green = np.array([35, 40, 0])
    upper_green = np.array([85, 255, 255])
    
    green_mask = cv2.inRange(hsv_image, lower_green, upper_green)
    
    v_channel[green_mask <= 0] = 0
    v_channel = cv2.equalizeHist(v_channel)
    v_channel[v_channel <= 230] = 0

    contours, _ = cv2.findContours(v_channel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Centre de l'image globale (concaténée)
    center_image_x = image.shape[1] / 2.0

    big_contours = []
    for contour in contours:
        epsilon = 0.04 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Filtre de taille et de forme (>= 2 points et périmètre > 15)
        if len(approx) >= 2 and cv2.arcLength(approx, True) > 15:
            
            # Position X moyenne du contour détecté
            contour_center_x = np.mean(approx[:, 0, 0])
            
            # Calcul de la distance relative au centre
            # Si le contour est à gauche du centre (oeil gauche), la distance sera négative.
            # S'il est à droite (oeil droit), elle sera positive.
            distance = contour_center_x - center_image_x
            
            # Taille verticale du contour
            size = approx[:, 0, 1].max() - approx[:, 0, 1].min()
            
            big_contours.append([approx, distance, size])

    return big_contours



def detect_triangles(image, eye="left"):
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
    v_channel[v_channel <= 230] = 0
    #remove small blobs
    #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    #v_channel = cv2.morphologyEx(v_channel, cv2.MORPH_OPEN, kernel)
    #v_channel = cv2.morphologyEx(v_channel, cv2.MORPH_CLOSE, kernel)

    #detecte les triangles 
    contours, _ = cv2.findContours(v_channel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    plot_image = cv2.cvtColor(v_channel, cv2.COLOR_GRAY2RGB)

    big_contours = []
    for contour in contours:
        epsilon = 0.04 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) >= 2 and cv2.arcLength(approx, True) > 15:  # Si le contour a 3 sommets, c'est un triangle
            
            #calculate the distance between the contour and the left or right edge of the image
            if eye == "right": 
                distance = approx[:, 0, 0].min()  # Distance au bord gauche
            else:
                distance = plot_image.shape[1] - approx[:, 0, 0].max()  # Distance au bord droit

            #size is the vertical size of the contour
            
            if len(approx) >= 2:
                size = approx[:, 0, 1].max() - approx[:, 0, 1].min()
            else:
                size = 0
            big_contours.append([approx, distance, size])

    return big_contours

def contours_to_aversive_odor(contours):
    """
    Transforme les contours de l'image globale en un signal 'olfactif' aversif 
    de dimension (4, 1).
    """
    fake_odor = np.zeros((4, 1))
    
    for contour in contours:
        # contour = [approx, distance, size]
        distance = contour[1]
        size = contour[2]
        
        if size > 10:  # Filtre de taille minimale
            if distance < 0:
                fake_odor[0, 0] += size
            else:
                fake_odor[1, 0] += size
                
    return fake_odor

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
        # Récupération des capteurs
        olfaction = sim.get_olfaction(sim.fly.name)
        vision = sim.get_raw_vision(sim.fly.name)
        
        # Concaténation des deux yeux : vision[0] (gauche) collé à gauche de vision[1] (droit)
        combined_vision = np.hstack((vision[0], vision[1]))
        
        # Détection sur l'image globale
        contours = detect_triangles_combined(combined_vision)

        fake_aversive_odor = contours_to_aversive_odor(contours)
        #print(fake_aversive_odor)
        combined_signals = np.hstack((olfaction, fake_aversive_odor))
        
        
        # 3. Calculer les drives
        drives = odor_intensity_to_control_signal(combined_signals)

        # 4. Appliquer au contrôleur moteur
        joint_angles, adhesion = self.turning_controller.step(drives)
        return joint_angles, adhesion
