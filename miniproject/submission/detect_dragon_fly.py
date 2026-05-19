import cv2
import numpy as np



def detecter_libellule_rouge(frame, surface_min=50, surface_max=10000):
    """
    Détecte la présence d'un objet rouge (libellule) dans une image RGB et retourne ses contours.
    Adapté pour les images générées par simulation.
    """
    # 1. Sécurité : Convertir en uint8 (0-255) si le simulateur renvoie des floats (0.0-1.0)
    if frame.dtype != np.uint8:
        if frame.max() <= 1.0:
            frame = (frame * 255).astype(np.uint8)
        else:
            frame = frame.astype(np.uint8)

    # 2. CRUCIAL : Convertir de RGB à HSV (et non plus BGR à HSV)
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    
    # 3. Définir les plages de la couleur rouge en HSV
    lower_red_1 = np.array([0, 100, 50])
    upper_red_1 = np.array([10, 255, 255])
    
    lower_red_2 = np.array([170, 100, 50])
    upper_red_2 = np.array([180, 255, 255])
    
    # 4. Créer les masques pour isoler les pixels rouges
    mask1 = cv2.inRange(hsv, lower_red_1, upper_red_1)
    mask2 = cv2.inRange(hsv, lower_red_2, upper_red_2)
    
    # Combiner les deux masques
    mask = cv2.bitwise_or(mask1, mask2)
    
    # 5. Nettoyage morphologique pour enlever le bruit
    kernel = np.ones((21, 21), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # 6. Trouver les contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    contours_valides = []
    
    # 7. Filtrer par surface
    for contour in contours:
        surface = cv2.contourArea(contour)
        if surface_min < surface < surface_max:
            contours_valides.append(contour)
            
    # Vérifier la présence
    presence = len(contours_valides) > 0
    
    return presence, contours_valides

