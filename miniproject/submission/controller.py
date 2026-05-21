from turtle import distance

import numpy as np
from torch import dist
from miniproject.simulation import MiniprojectSimulation
from submission.vision_model import FlyVisionModel
from submission.detect_dragonfly import detect_dragonfly
import cv2
from collections import deque


def contour_to_aversive_odor(best_contour, olfaction, grass_in_middle, direction):
    """
    Transforme l'unique meilleur contour en un signal 'olfactif' aversif.
    """
    fake_odor = np.zeros((4, 1))

    if best_contour is not None:
        # best_contour = [approx, distance, size]
        distance = best_contour[1]
        size = best_contour[2]

        attractive_intensities = np.average(
            olfaction.reshape(2, 2), axis=0, weights=[9, 1]
        )

        if grass_in_middle == True:
            if direction == None:
                if attractive_intensities[0] - attractive_intensities[1] > 0:
                    fake_odor[1, 0] += size
                    direction = "left"
                else:
                    fake_odor[0, 0] += size
                    direction = "right"
            else:
                if direction == "left":
                    fake_odor[1, 0] += size
                else:
                    fake_odor[0, 0] += size

        else:
            if distance < 0:
                # Oeil gauche détecté -> stimuler le capteur aversif gauche
                fake_odor[0, 0] += size
            else:
                # Oeil droit détecté -> stimuler le capteur aversif droit
                fake_odor[1, 0] += size

    return fake_odor, direction


def odor_intensity_to_control_signal(
    odor_intensities,
    state,
    aversive_gain,   #1800
    attractive_gain=-30,  #-3500
    attractive_gain_hrc = 300*5,
    hrc_signal = 0.0
):
    """Convert odor sensor readings to a turning control signal."""

    attractive_intensities = np.average(
        odor_intensities[:, 0].reshape(2, 2), axis=0, weights=[9, 1]
    )
    # attractive_bias = (
    #     attractive_gain
    #     * (attractive_intensities[0] - attractive_intensities[1])
    #     / attractive_intensities.mean()
    #     if attractive_intensities.mean() != 0
    #     else 0
    # )

    # if ((attractive_intensities[0] - attractive_intensities[1]) / attractive_intensities.mean())>0.1:
    #     attractive_gain = attractive_gain / 1000

    # elif ((attractive_intensities[0] - attractive_intensities[1]) / attractive_intensities.mean())>0.01:
    #     attractive_gain = attractive_gain / 100

    # elif ((attractive_intensities[0] - attractive_intensities[1]) / attractive_intensities.mean())>0.001:
    #     attractive_gain = attractive_gain / 10


    # if np.abs(hrc_signal) < 0.0001:
    #     attractive_bias = (
    #         attractive_gain
    #         * (attractive_intensities[0] - attractive_intensities[1])
    #         / attractive_intensities.mean()
    #         if attractive_intensities.mean() != 0
    #         else 0
    #     )
    #     hrc_on = False
    # else:
    #     attractive_bias = hrc_signal * attractive_gain_hrc
    #     hrc_on = True

    attractive_bias = (
            attractive_gain
            * (attractive_intensities[0] - attractive_intensities[1])
            / attractive_intensities.mean()
            if attractive_intensities.mean() != 0
            else 0
        )
    hrc_on = False

    if odor_intensities.shape[1] > 1:
        aversive_intensities = np.average(
            odor_intensities[:, 1].reshape(2, 2), axis=0, weights=[10, 0]
        )
        # aversive_bias = (
        #     aversive_gain
        #     * (aversive_intensities[0] - aversive_intensities[1])
        #     / aversive_intensities.mean()
        #     if aversive_intensities.mean() != 0
        #     else 0
        # )
        aversive_bias = (
            aversive_gain * (aversive_intensities[0] - aversive_intensities[1]) / 512
        )

        # if aversive_intensities.mean() != 0:
        #     print((aversive_intensities[0] - aversive_intensities[1]) / aversive_intensities.mean() if aversive_intensities.mean() != 0 else 0)

        # # print(aversive_intensities)
        # # print(aversive_intensities.mean())
        # # print(aversive_bias)
        # if attractive_intensities.mean() != 0:
        #      print((attractive_intensities[0] - attractive_intensities[1]) / attractive_intensities.mean() if attractive_intensities.mean() != 0 else 0)
    else:
        aversive_bias = 0

    clipped_attractive_bias = np.clip(attractive_bias, -1, 1)
    effective_bias = aversive_bias + clipped_attractive_bias
    effective_bias_norm = np.tanh(effective_bias**2) * np.sign(effective_bias)
    assert np.sign(effective_bias_norm) == np.sign(effective_bias)

    if state == "ALIGN_WITH_FOOD":
        control_signal = np.zeros(2)
        side_to_modulate = int(effective_bias_norm > 0)
        modulation_amount = np.abs(effective_bias_norm)
        if side_to_modulate == 0:
            control_signal[side_to_modulate] -= 0.8
            control_signal[1] += 0.8
        else:
            control_signal[side_to_modulate] -= 0.8
            control_signal[0] += 0.8

    elif state == "GO_TO_FOOD":
        control_signal = np.ones(2)
        side_to_modulate = int(effective_bias_norm > 0)
        modulation_amount = np.abs(effective_bias_norm) * 0.8 
        if side_to_modulate == 0:
            control_signal[side_to_modulate] -= modulation_amount
            control_signal[1] += modulation_amount
        else:
            control_signal[side_to_modulate] -= modulation_amount
            control_signal[0] += modulation_amount

    elif state == "EVASION":
        control_signal = np.ones(2) * 1.8
        side_to_modulate = int(effective_bias_norm > 0)
        modulation_amount = np.abs(effective_bias_norm) * 0.8 * 1.8
        if side_to_modulate == 0:
            control_signal[side_to_modulate] -= modulation_amount
            control_signal[1] += modulation_amount
        else:
            control_signal[side_to_modulate] -= modulation_amount
            control_signal[0] += modulation_amount


    return control_signal, aversive_bias, clipped_attractive_bias, hrc_on

def compute_hrc(left_history, right_history, delay):

    if len(left_history) <= delay:
        return 0.0

    L_now = left_history[-1]
    R_now = right_history[-1]

    L_delayed = left_history[-1 - delay]
    R_delayed = right_history[-1 - delay]

    hrc = (
        L_delayed * R_now
        - R_delayed * L_now
    )/(L_delayed * R_now + R_delayed * L_now )  # Adding a small constant to avoid division by zero

    return hrc


class Controller:
    def __init__(self, sim: MiniprojectSimulation):
        # from submission.turning_controller import TurningController
        from flygym.examples.locomotion import TurningController
        self.turning_controller = TurningController(sim.timestep)
        self.step_count = 0
        self.best_contour = None
        self.contours = []
        self.dragonfly_contours = []

        self.vision_model = FlyVisionModel()

        self.state = "ALIGN_WITH_FOOD"
        self.previous_state = "ALIGN_WITH_FOOD"
        self.evasion_step = 0
        self.EVASION_MAX_STEP = 3000

        self.GRASS_IN_MIDDLE = False
        self.grass_step = 0
        self.direction = None

        self.olf_left_history = deque(maxlen=100)
        self.olf_right_history = deque(maxlen=100)

        self.hrc_delay = 1
        self.hrc_signal = 0.0

        self.hrc_on = False
        self.aversive_bias = 0.0
        self.attractive_bias = 0.0


    def step(self, sim: MiniprojectSimulation):
        olfaction = sim.get_olfaction(sim.fly.name)

        if self.step_count%200 == 0:

            #get raw vision and combine both eyes
            vision = sim.get_raw_vision(sim.fly.name)
            combined_vision = np.hstack((vision[0], vision[1]))

            #detect dragonfly and change state accordingly
            dragonfly, self.dragonfly_contours = detect_dragonfly(combined_vision)
            if dragonfly :
                #store previous state to come back to it later
                if self.evasion_step == 0 :
                    self.previous_state = self.state
                    self.state = "EVASION"

            if self.state == "GO_TO_FOOD" or self.state == "EVASION":
                #detect grass contours
                self.contours = self.vision_model.detect_grass(combined_vision)
                # print(self.contours)
                #find the grass with the highest score to avoid it
                best_score = -1
                self.best_contour = None

                for contour in self.contours:
                    score = contour[3]
                    size = contour[2]
                    # print(contour)
                    # print(size)
                    if size > 150:
                        if score > best_score:
                            best_score = score
                            self.best_contour = contour

                if self.best_contour is not None:
                    if abs(self.best_contour[1]) < 40:
                        self.GRASS_IN_MIDDLE = True
                    else :
                        self.GRASS_IN_MIDDLE = False
                
                if self.GRASS_IN_MIDDLE == True :
                    self.grass_step = self.step_count
                    #self.KEEP_DIRECTION = True
                if self.GRASS_IN_MIDDLE == False :
                    if (self.step_count - self.grass_step) < 1000 and self.step_count > 1000 :
                        self.GRASS_IN_MIDDLE = True
                    else :
                        self.direction = None
                        #self.KEEP_DIRECTION = False


        if self.state == "ALIGN_WITH_FOOD":
                signals = olfaction
                gain_aversive = 0

                attractive_intensities = np.average(
                    olfaction.reshape(2, 2), axis=0, weights=[9, 1]
                )

                if (attractive_intensities[0]/attractive_intensities[1]) > 0.99 and (attractive_intensities[0]/attractive_intensities[1]) < 1.01:
                    print("Attractive intensities left: ", attractive_intensities[0])
                    print("Attractive intensities right: ", attractive_intensities[1])
                    self.state = "GO_TO_FOOD"

        elif self.state == "GO_TO_FOOD":

                
                fake_aversive_odors, self.direction = contour_to_aversive_odor(self.best_contour,olfaction, self.GRASS_IN_MIDDLE, self.direction)
                signals = np.hstack((olfaction, fake_aversive_odors))
                gain_aversive = 2

        elif self.state == "EVASION":
            signals = olfaction
            gain_aversive = 0
            if self.evasion_step < self.EVASION_MAX_STEP: #and len(self.recent_drives) > 0
                self.evasion_step += 1
            else:
                #go back to previous state
                self.state = self.previous_state
                self.evasion_step = 0

        attractive_intensities = np.average(
                    olfaction.reshape(2, 2), axis=0, weights=[9, 1]
                )

        self.olf_left_history.append(attractive_intensities[0])
        self.olf_right_history.append(attractive_intensities[1])

        self.hrc_signal = compute_hrc(self.olf_left_history, self.olf_right_history, self.hrc_delay)

        if self.direction is not None:
            gain_attractive = 0
        else:
            gain_attractive = -30

        drives, self.aversive_bias, self.attractive_bias, self.hrc_on = odor_intensity_to_control_signal(signals,self.state, aversive_gain=gain_aversive, attractive_gain=gain_attractive, hrc_signal=self.hrc_signal)
        # print("state au premier step: ", self.state)
        # print("olfaction au premier step: ", olfaction)
        # print("attractive_intensities au premier step: ", attractive_intensities)

        # Génération des commandes

        # drives = np.array([-1, 1]) 


        # debuggage
        #drives = np.zeros(2)
        forces = sim.get_external_force(sim.fly.name, subtract_adhesion_force = True)
        joint_angles, adhesion = self.turning_controller.step(drives)


        self.step_count += 1
        # On retourne maintenant 5 éléments, incluant le best_contour
        return joint_angles, adhesion,self.contours, self.best_contour, self.dragonfly_contours, self.state, drives, forces, attractive_intensities