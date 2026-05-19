import numpy as np
from torch import dist
from miniproject.simulation import MiniprojectSimulation
from submission.vision_model import FlyVisionModel
from submission.detect_dragonfly import detect_dragonfly
import cv2

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

    elif state == "EVASION":
        control_signal = np.ones(2)*-1.3


    return control_signal


class Controller:
    def __init__(self, sim: MiniprojectSimulation):
        from submission.turning_controller import TurningController
        self.turning_controller = TurningController(sim.timestep)
        self.step_count = 0
        self.best_contour = None
        self.contours = []
        self.dragonfly_contours = []

        self.vision_model = FlyVisionModel()

        self.state = "ALIGN_WITH_FOOD"
        self.previous_state = "ALIGN_WITH_FOOD"
        self.evasion_step = 0
        self.EVATION_MAX_STEP = 3000


    def step(self, sim: MiniprojectSimulation):
        olfaction = sim.get_olfaction(sim.fly.name)

        if self.step_count%70 == 0:

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

            if self.state == "GO_TO_FOOD":

                #detect grass contours
                self.contours = self.vision_model.detect_grass(combined_vision)

                #find the grass with the highest score to avoid it
                best_score = -1
                self.best_contour = None

                for contour in self.contours:
                    score = contour[3]
                    size = contour[2]

                    if size > 190:  
                        if score > best_score:
                            best_score = score
                            self.best_contour = contour
                


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
                fake_aversive_odors = contour_to_aversive_odor(self.best_contour)

                signals = np.hstack((olfaction, fake_aversive_odors))

                gain_aversive = 1600

                

        elif self.state == "EVASION":
            signals = olfaction
            gain_aversive = 0
            if self.evasion_step < self.EVATION_MAX_STEP: #and len(self.recent_drives) > 0
                self.evasion_step += 1
            else:
                #go back to previous state
                self.state = self.previous_state
                self.evasion_step = 0


        drives = odor_intensity_to_control_signal(signals,self.state, aversive_gain=gain_aversive)
        # print("state au premier step: ", self.state)
        # print("olfaction au premier step: ", olfaction)
        # print("attractive_intensities au premier step: ", attractive_intensities)

        # Génération des commandes
        

        # debuggage
        #drives = np.zeros(2)
        forces = sim.get_external_force(sim.fly.name, subtract_adhesion_force = True)
        joint_angles, adhesion = self.turning_controller.step(drives, forces)


        self.step_count += 1
        # On retourne maintenant 5 éléments, incluant le best_contour
        return joint_angles, adhesion,self.contours, self.best_contour, self.dragonfly_contours, self.state, drives, forces 