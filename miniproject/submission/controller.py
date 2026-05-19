import numpy as np
from torch import dist
from miniproject.simulation import MiniprojectSimulation
from submission.vision_model import FlyVisionModel
import cv2

def detect_dragonfly(img):
    dragonfly_detected = False

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])
    mask = cv2.inRange(hsv, lower_red, upper_red)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        dragonfly_detected = True


    return dragonfly_detected


def detect_triangles_combined(image):
    """
    Détecte les triangles sur une image concaténée (gauche + droite).
    La distance retournée est négative si le triangle est à gauche, positive s'il est à droite.
    """

    scale = 1.0
    width = int(image.shape[1] * scale)
    height = int(image.shape[0] * scale)
    small_img = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    hsv_image = cv2.cvtColor(small_img, cv2.COLOR_RGB2HSV)

    # Extraire le canal V
    v_channel = hsv_image[:, :, 0].copy()
    v_channel2 = hsv_image[:, :, 2].copy()
    v_channel3 = hsv_image[:, :, 0].copy()

    # Identifier les pixels verts
    lower_green = np.array([35, 40, 0])
    upper_green = np.array([85, 255, 255])
    green_mask = cv2.inRange(hsv_image, lower_green, upper_green)

    lower_brown = np.array([12, 100, 20])
    upper_brown = np.array([40, 255, 200])
    brown_mask = cv2.inRange(hsv_image, lower_brown, upper_brown)

    # Soustraire le vert du canal V
    v_channel[green_mask <= 0] = 0
    v_channel2[green_mask <= 0] = 0
    v_channel3[brown_mask <= 0] = 0
    v_channel3[v_channel3 > 0] = 255



    #kernel part 1
    kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (80, 80))
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
    kernel4 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (81, 81))

    #part1
    v_channel[v_channel > 0] = 255
    v_channel_eroded = cv2.morphologyEx(v_channel, cv2.MORPH_ERODE, kernel1)
    v_channel3_dilated = cv2.morphologyEx(v_channel3, cv2.MORPH_DILATE, kernel2)
    v_channel_all = cv2.add(v_channel_eroded, v_channel3_dilated)
    v_channel_all[v_channel_all > 0] = 255
    v_channel_all_dilated = cv2.morphologyEx(v_channel_all, cv2.MORPH_DILATE, kernel4)
    #v_channel = v_channel2 + v_channel
    v_channel[v_channel_all_dilated > 0] = 0

    #part 2


    v_channel2[green_mask <= 0] = 0
     #get left side mean and right side mean
    left_mean = np.mean(v_channel2[:, :v_channel2.shape[1]//2])
    right_mean = np.mean(v_channel2[:, v_channel2.shape[1]//2:])
    #decoupe en deux
    left_side = v_channel2[:, :v_channel2.shape[1]//2]
    right_side = v_channel2[:, v_channel2.shape[1]//2:]

    left_side[left_side < left_mean+40] = left_mean+40
    right_side[right_side < right_mean+40] = right_mean+40

    #v_channel = np.hstack((left_side, right_side))
    left_side = cv2.equalizeHist(left_side)
    right_side = cv2.equalizeHist(right_side)
    v_channel2 = np.hstack((left_side, right_side))
    v_channel2[v_channel2<= 240] = 0
    v_channel2[v_channel2 > 0] = 255


    v_channel = cv2.bitwise_or(v_channel, v_channel2)

    # Détecte les contours
    contours, _ = cv2.findContours(v_channel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Centre de l'image globale
    center_image_x = small_img.shape[1] / 2.0


    big_contours = []
    for contour in contours:
        epsilon = 0.04 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) >= 2 and cv2.arcLength(approx, True) > 15:
            contour_center_x = np.mean(approx[:, 0, 0])
            distance = (contour_center_x - center_image_x)/scale
            size = (approx[:, 0, 1].max() - approx[:, 0, 1].min())/scale
            if abs(distance) < 20 :
                distance = abs(distance)/scale

            if distance == 0 :
                #avoid division by zero
                score = np.inf
            else :
                score = size / (abs(distance)*0.5)

            big_contours.append([approx , distance , size, score])



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
    state,
    attractive_gain=-3500,
    aversive_gain=1800
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

    if state == "ALIGN_WITH_FOOD":
        control_signal = np.zeros(2)
        side_to_modulate = int(effective_bias_norm > 0)
        modulation_amount = np.abs(effective_bias_norm)
        if side_to_modulate == 0:
            control_signal[side_to_modulate] -= modulation_amount
            control_signal[1] += modulation_amount
        else:
            control_signal[side_to_modulate] -= modulation_amount
            control_signal[0] += modulation_amount

    elif state == "GO_TO_FOOD":
        control_signal = np.ones(2)*1.4
        side_to_modulate = int(effective_bias_norm > 0)
        modulation_amount = np.abs(effective_bias_norm) * 0.8 * 1.4
        if side_to_modulate == 0:
            control_signal[side_to_modulate] -= modulation_amount
            control_signal[1] += modulation_amount
        else:
            control_signal[side_to_modulate] -= modulation_amount
            control_signal[0] += modulation_amount

    return control_signal


class Controller:
    def __init__(self, sim: MiniprojectSimulation):
        from flygym.examples.locomotion import TurningController
        self.turning_controller = TurningController(sim.timestep)
        self.step_count = 0
        self.best_contour = None
        self.contours = []
        self.vision_model = FlyVisionModel()
        self.state = "ALIGN_WITH_FOOD"


    def step(self, sim: MiniprojectSimulation):
        olfaction = sim.get_olfaction(sim.fly.name)


        if self.step_count%70 == 0:
            if self.state == "GO_TO_FOOD":

                vision = sim.get_raw_vision(sim.fly.name)
                combined_vision = np.hstack((vision[0], vision[1]))

                # Utiliser le modèle de vision pour détecter l'herbe
                self.contours = self.vision_model.detect_grass(combined_vision)

                if detect_dragonfly(combined_vision):
                    print("Dragonfly detected!")

                best_score = -1
                self.best_contour = None

                for contour in self.contours:
                    score = contour[3]
                    size = contour[2]

                    if size > 200:  # Filtre de taille minimale
                        if score > best_score:
                            best_score = score
                            self.best_contour = contour
                # ---------------------------------------------------------


        if self.state == "ALIGN_WITH_FOOD":
                signals = olfaction
                gain_aversive = 0

                attractive_intensities = np.average(
                    olfaction.reshape(2, 2), axis=0, weights=[9, 1]
                )
                # if abs(attractive_intensities[0] - attractive_intensities[1]) < 0.1e-8:
                #     print("Attractive intensities left: ", attractive_intensities[0])
                #     print("Attractive intensities right: ", attractive_intensities[1])
                #     self.state = "GO_TO_FOOD"

                if (attractive_intensities[0]/attractive_intensities[1]) > 0.99 and (attractive_intensities[0]/attractive_intensities[1]) < 1.01:
                    print("Attractive intensities left: ", attractive_intensities[0])
                    print("Attractive intensities right: ", attractive_intensities[1])
                    self.state = "GO_TO_FOOD"

        elif self.state == "GO_TO_FOOD":
                fake_aversive_odors = contour_to_aversive_odor(self.best_contour)

                signals = np.hstack((olfaction, fake_aversive_odors))

                gain_aversive = 1200

        # print("state au premier step: ", self.state)
        # print("olfaction au premier step: ", olfaction)
        # print("attractive_intensities au premier step: ", attractive_intensities)

        # Génération des commandes
        drives = odor_intensity_to_control_signal(signals,self.state, aversive_gain=gain_aversive)

        # debuggage
        #drives = np.zeros(2)

        joint_angles, adhesion = self.turning_controller.step(drives)


        self.step_count += 1
        # On retourne maintenant 5 éléments, incluant le best_contour
        return joint_angles, adhesion,self.contours, self.best_contour