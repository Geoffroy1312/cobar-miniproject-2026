import numpy as np
from miniproject.simulation import MiniprojectSimulation
import cv2
import matplotlib.pyplot as plt

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
    v_channel[v_channel <= 250] = 0
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

    # for contour in big_contours:
    #     #affiche les contour de l'image dans la couleur de la distance (plus c'est rouge plus c'est proche du bord)
    #     valeur_rouge = 255 - int(contour[2] * 2550 / plot_image.shape[0])
    #     cv2.drawContours(plot_image, [contour[0]], -1, (valeur_rouge, 0, 0), 2)
    # imgplot = plt.imshow(plot_image)
    # plt.show()
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
        contours_left = detect_triangles(vision_left, eye="left")
        contours_right = detect_triangles(vision_right, eye="right")



        drives_from_olfaction = odor_intensity_to_control_signal(olfaction)
        # print(f"odor control signal: {drives}")

        max_ratio_right = 0
        max_ratio_left = 0

        drives = np.zeros(2)

        #give a penatly to the control signal if there is a triangle detected in the left eye and a bonus if there is a triangle detected in the right eye
        for contour in contours_right:
            # print("contour right hauteur: ", contour[2])
            # print("contour right distance: ", contour[1])
            if contour[2] > 100: #if the triangle is big enough
                if contour[1] < vision_right.shape[1] / 4: #if the triangle is closer to the right
                    ratio = contour[2]/contour[1] if contour[1] != 0 else 0
                    # print("ratio right: ", ratio)
                    if ratio > max_ratio_right:
                        max_ratio_right = ratio
                        distance_right = contour[1]
                        # print("max ratio right: ", max_ratio_right)
                    # print("vision_right_shape_0 and 1: ", vision_right.shape[0], vision_right.shape[1])
                    # print("contour_1 (distance): ", contour[1])
                    # print("contour_2 (hauteur): ", contour[2])
                    # print("right drives: ", drives)

                #print("triangle detected in right eye, applying bonus to left drive")
        for contour in contours_left:
            # print("contour left hauteur: ", contour[2])
            # print("contour left distance: ", contour[1])
            if contour[2] > 100: #if the triangle is big enough
                if contour[1] < vision_left.shape[1] / 4: #if the triangle is closer to the left in the left eye
                    ratio = contour[2]/contour[1] if contour[1] != 0 else 0
                    # print("ratio left: ", ratio)
                    if ratio > max_ratio_left:
                        max_ratio_left = ratio
                        distance_left = contour[1]
                        # print("max ratio left: ", max_ratio_left)
                    # print("vision_left_shape_0 and 1: ", vision_left.shape[0], vision_left.shape[1])
                    # print("contour_1 (distance): ", contour[1])
                    # print("contour_2 (hauteur): ", contour[2])
                    # print("left drives: ", drives)
                #print("triangle detected in left eye, applying bonus to right drive")

        # print("max ratio right: ", max_ratio_right)
        # print("max ratio left: ", max_ratio_left)
        # if max_ratio_right > max_ratio_left:
        #     drives[1] = -1*(vision_right.shape[0] - distance_right) / vision_right.shape[0] #closer to the right in right eye gives bonus to right drive
        #     drives[0] = 1*(vision_right.shape[0] - distance_right) / vision_right.shape[0] #closeer to the left in right eye gives bonus to left drive
        #     turning_right = True
        #     turning_left = False
        # elif max_ratio_left > max_ratio_right:
        #     drives[0] = -1*(vision_left.shape[0] - distance_left) / vision_left.shape[0] #closer to the left in left eye gives bonus to left drive
        #     drives[1] = 1*(vision_left.shape[0] - distance_left) / vision_left.shape[0] #closer to the right in left eye gives bonus to right drive
        #     turning_left = True
        #     turning_right = False
        # else:
        #     drives = drives_from_olfaction
        #     turning_left = False
        #     turning_right = False
        
        if max_ratio_right > max_ratio_left:
            distance = distance_right
        elif max_ratio_left > max_ratio_right:
            distance = distance_left

        # print("drives_from_olfaction: ", drives_from_olfaction)

        

        if max_ratio_right > 0 or max_ratio_left > 0:
            if drives_from_olfaction[0] < (drives_from_olfaction[1] - 0.2): #bigger drive on the right => turn left
                drives[1] = 1*(vision_left.shape[1] - distance) / vision_left.shape[0] #bonus for right drive
                drives[0] = -1*(vision_left.shape[1] - distance) / vision_left.shape[0] #penalty for left drive
                turning_right = False
                turning_left = True
            elif (drives_from_olfaction[0] - 0.2) > drives_from_olfaction[1]:
                # print(drives_from_olfaction)
                drives[0] = 1*(vision_right.shape[1] - distance) / vision_right.shape[0] #bonus for left drive
                drives[1] = -1*(vision_right.shape[1] - distance) / vision_right.shape[0] #penalty for right drive
                turning_left = False
                turning_right = True
            else: #if the food is more or less in front, always turn right
                # print("food is more or less in front, always turn left")
                drives[0] = -1*(vision_left.shape[1] - distance) / vision_left.shape[0] #bonus for right drive
                drives[1] = 1*(vision_left.shape[1] - distance) / vision_left.shape[0] #penalty for left drive
                turning_left = False
                turning_right = False

        else:
            drives = drives_from_olfaction
            turning_left = False
            turning_right = False

        joint_angles, adhesion = self.turning_controller.step(drives)
        return joint_angles, adhesion, turning_right, turning_left
