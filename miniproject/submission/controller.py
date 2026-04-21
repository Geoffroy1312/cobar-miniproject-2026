import numpy as np
from miniproject.simulation import MiniprojectSimulation


class Controller:
    def __init__(self, sim: MiniprojectSimulation):
        # you may also implement your own turning controller
        from flygym.examples.locomotion import TurningController

        self.turning_controller = TurningController(sim.timestep)

    def step(self, sim: MiniprojectSimulation):
        # implement your control algorithm here
        olfaction = sim.get_olfaction(sim.fly.name)

        # get other observations as needed
        vision = sim.get_vision(sim.fly.name)
        proprioception = sim.get_proprioception(sim.fly.name)
        print("Olfaction:", olfaction)
        print("Vision:", vision)
        print("Proprioception:", proprioception)

        drive_left = 0.0
        drive_right = 0.0
        drives = np.array([drive_left, drive_right])  # replace with your control logic
        joint_angles, adhesion = self.turning_controller.step(drives)
        return joint_angles, adhesion
