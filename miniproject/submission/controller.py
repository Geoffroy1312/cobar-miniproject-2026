import numpy as np
from miniproject.simulation import MiniprojectSimulation
import cv2

def detect_triangles_combined(image):
    """
    Détecte les triangles sur une image concaténée (gauche + droite).
    La distance retournée est négative si le triangle est à gauche, positive s'il est à droite.
    """
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # Extraire le canal V
    v_channel = hsv_image[:, :, 2].copy()

    # Identifier les pixels verts
    lower_green = np.array([35, 40, 0])
    upper_green = np.array([85, 255, 255])
    green_mask = cv2.inRange(hsv_image, lower_green, upper_green)

    # Soustraire le vert du canal V
    v_channel[green_mask <= 0] = 0
    v_channel = cv2.equalizeHist(v_channel)
    v_channel[v_channel <= 250] = 0

    # Détecte les contours
    contours, _ = cv2.findContours(v_channel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Centre de l'image globale
    center_image_x = image.shape[1] / 2.0

    big_contours = []
    for contour in contours:
        epsilon = 0.04 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        if len(approx) >= 2 and cv2.arcLength(approx, True) > 15:
            contour_center_x = np.mean(approx[:, 0, 0])
            distance = contour_center_x - center_image_x
            size = approx[:, 0, 1].max() - approx[:, 0, 1].min()
            
            big_contours.append([approx, distance, size])

    return big_contours


def contour_to_aversive_odor(best_contour):
    """
    Transforme l'unique meilleur contour en un signal 'olfactif' aversif.
    """
    fake_odor = np.zeros((4, 1))
    
    if best_contour is not None:
        # best_contour = [approx, distance, size]
        distance = best_contour[1]
        size = best_contour[2]
        
        if distance < 0:
            # Oeil gauche détecté -> stimuler le capteur aversif gauche
            fake_odor[0, 0] += size
        else:
            # Oeil droit détecté -> stimuler le capteur aversif droit
            fake_odor[1, 0] += size
                
    return fake_odor


def odor_intensity_to_control_signal(
    odor_intensities,
    attractive_gain=-1000,
    aversive_gain=20,
):
    """Convert odor sensor readings to a turning control signal."""

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

    control_signal = np.ones(2)*2.0
    side_to_modulate = int(effective_bias_norm > 0)
    modulation_amount = np.abs(effective_bias_norm) * 1.6
    control_signal[side_to_modulate] -= modulation_amount

    return control_signal


class Controller:
    def __init__(self, sim: MiniprojectSimulation):
        from flygym.examples.locomotion import TurningController
        self.turning_controller = TurningController(sim.timestep)

    def step(self, sim: MiniprojectSimulation):
        olfaction = sim.get_olfaction(sim.fly.name)
        vision = sim.get_raw_vision(sim.fly.name)
        
        # Concaténation des deux yeux
        combined_vision = np.hstack((vision[0], vision[1]))
        
        # Détection de tous les contours
        contours = detect_triangles_combined(combined_vision)

        # ---------------------------------------------------------
        # RECHERCHE DU MEILLEUR CONTOUR (Le plus haut et le plus au milieu)
        # ---------------------------------------------------------
        best_contour = None
        best_score = -1
        
        for contour in contours:
            distance = contour[1]
            size = contour[2]
            
            if size > 100:  # Filtre de taille minimale
                # Score = ratio taille / distance au centre. 
                # Le +1.0 évite une division par zéro si le contour est parfaitement centré.
                if distance == 0 :
                    #avoid division by zero 
                    best_contour = contour
                else :
                    score = size / abs(distance)
                
                if score > best_score:
                    best_score = score
                    best_contour = contour
        # ---------------------------------------------------------

        # Traduction du meilleur contour en odeur aversive
        fake_aversive_odor = contour_to_aversive_odor(best_contour)

        # Fusion des signaux
        if olfaction.shape[1] == 1:
            combined_signals = np.hstack((olfaction, fake_aversive_odor))
        else:
            combined_signals = olfaction.copy()
            combined_signals[:, 1:2] += fake_aversive_odor

        # Génération des commandes 
        drives = odor_intensity_to_control_signal(combined_signals)

        turning_left = bool(drives[0] < 1.0)
        turning_right = bool(drives[1] < 1.0)

        joint_angles, adhesion = self.turning_controller.step(drives)
        
        # On retourne maintenant 5 éléments, incluant le best_contour
        return joint_angles, adhesion, turning_right, turning_left, best_contour