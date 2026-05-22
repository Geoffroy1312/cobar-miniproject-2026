from turtle import distance

import numpy as np
from torch import dist
from miniproject.simulation import MiniprojectSimulation
from submission.vision_model import FlyVisionModel
from submission.detect_dragonfly import detect_dragonfly
from submission.get_control_signals import contour_to_aversive_odor, odor_intensity_to_control_signal

class Controller:
    def __init__(self, sim: MiniprojectSimulation):
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
                
                #find the grass with the highest score to avoid it
                best_score = -1
                self.best_contour = None
                for contour in self.contours:
                    score = contour[3]
                    size = contour[2]
                    if size > 150:
                        if score > best_score:
                            best_score = score
                            self.best_contour = contour

                #grass in the middle logic
                if self.best_contour is not None:
                    if abs(self.best_contour[1]) < 40:
                        self.GRASS_IN_MIDDLE = True
                    else :
                        self.GRASS_IN_MIDDLE = False
                
                if self.GRASS_IN_MIDDLE == True :
                    self.grass_step = self.step_count
                if self.GRASS_IN_MIDDLE == False :
                    if (self.step_count - self.grass_step) < 1000 and self.step_count > 1000 :
                        self.GRASS_IN_MIDDLE = True
                    else :
                        self.direction = None


        if self.state == "ALIGN_WITH_FOOD":
                signals = olfaction
                gain_aversive = 0

                attractive_intensities = np.average(
                    olfaction.reshape(2, 2), axis=0, weights=[9, 1]
                )

                #once the fly is at the right angle, change state
                if (attractive_intensities[0]/attractive_intensities[1]) > 0.99 and (attractive_intensities[0]/attractive_intensities[1]) < 1.01:
                    self.state = "GO_TO_FOOD"

        elif self.state == "GO_TO_FOOD":
                fake_aversive_odors, self.direction = contour_to_aversive_odor(self.best_contour,olfaction, self.GRASS_IN_MIDDLE, self.direction)
                signals = np.hstack((olfaction, fake_aversive_odors))
                gain_aversive = 2

        elif self.state == "EVASION":
            fake_aversive_odors, self.direction = contour_to_aversive_odor(self.best_contour,olfaction, self.GRASS_IN_MIDDLE, self.direction)
            signals = np.hstack((olfaction, fake_aversive_odors))
            gain_aversive = 2
            
            if self.evasion_step < self.EVASION_MAX_STEP: 
                self.evasion_step += 1
            else:
                #go back to previous state
                self.state = self.previous_state
                self.evasion_step = 0

        attractive_intensities = np.average(
                    olfaction.reshape(2, 2), axis=0, weights=[9, 1]
                )

        if self.direction is not None:
            gain_attractive = 0
        else:
            gain_attractive = -30

        drives, self.aversive_bias, self.attractive_bias = odor_intensity_to_control_signal(signals,self.state, aversive_gain=gain_aversive, attractive_gain=gain_attractive)
        
        joint_angles, adhesion = self.turning_controller.step(drives)

        self.step_count += 1
        
        return joint_angles, adhesion #,self.contours, self.best_contour, self.dragonfly_contours, self.state, drives, attractive_intensities