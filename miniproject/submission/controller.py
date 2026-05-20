import numpy as np
from miniproject.simulation import MiniprojectSimulation
from submission.vision_model import FlyVisionModel
from submission.detect_dragonfly import detect_dragonfly
import cv2

import numpy as np
import cv2
from miniproject.simulation import MiniprojectSimulation
from submission.vision_model import FlyVisionModel
from submission.detect_dragonfly import detect_dragonfly

class Controller:
    def __init__(self, sim: MiniprojectSimulation):
        from submission.turning_controller import TurningController
        self.turning_controller = TurningController(sim.timestep)
        self.step_count = 0
        
        # Attributs conservés pour la compatibilité stricte du retour
        self.contours = []
        self.best_contour = None
        self.dragonfly_contours = []

        self.vision_model = FlyVisionModel()

        # Retour à votre logique de démarrage : on s'aligne d'abord
        self.state = "ALIGN_WITH_FOOD"
        self.previous_state = "ALIGN_WITH_FOOD"
        self.evasion_step = 0
        self.EVATION_MAX_STEP = 3000
        self.smooth_bias = 0.0
        
        self.obstacle_cost = None

    def step(self, sim: MiniprojectSimulation):
        olfaction = sim.get_olfaction(sim.fly.name)

        # 1. Mise à jour de la vision et des obstacles (tous les 70 pas)
        if self.step_count % 70 == 0:
            vision = sim.get_raw_vision(sim.fly.name)
            combined_vision = np.hstack((vision[0], vision[1]))

            dragonfly, self.dragonfly_contours = detect_dragonfly(combined_vision)
            if dragonfly:
                if self.evasion_step == 0:
                    self.previous_state = self.state
                    self.state = "EVASION"

            occupancy_1d = self.vision_model.get_1d_occupancy_grid(combined_vision)
            
            kernel_size = 181 
            blurred = cv2.GaussianBlur(
                occupancy_1d.astype(np.float32).reshape(1, -1), 
                (kernel_size, 1), 
                0
            ).flatten()
            
            if blurred.max() > 0:
                blurred /= blurred.max()
            self.obstacle_cost = blurred

        if self.obstacle_cost is None:
            self.obstacle_cost = np.zeros(sim.get_raw_vision(sim.fly.name)[0].shape[1] * 2)

        # 2. Lecture des capteurs olfactifs
        attractive_intensities = np.average(olfaction.reshape(2, 2), axis=0, weights=[9, 1])
        left_olf = attractive_intensities[0]
        right_olf = attractive_intensities[1]
        total_olf = left_olf + right_olf

        if total_olf > 1e-50:
            raw_bias = (left_olf - right_olf) 
            sensibilite_odeur = 10.0 
            inst_bias = raw_bias * sensibilite_odeur # Biais instantané
            
            # Zone morte sur le biais instantané
            if abs(inst_bias) < 0.15:
                inst_bias = 0.0
                
            inst_bias = np.clip(inst_bias, -1.0, 1.0)
            
            # --- NOUVEAU : Filtre Passe-Bas (Lissage Exponentiel) ---
            # alpha = 0.1 signifie que le biais final est composé de 10% de la nouvelle mesure
            # et de 90% de l'ancienne position. Ajustez vers 0.05 pour lisser encore plus.
            alpha = 0.1 
            if not hasattr(self, 'smooth_bias'):
                self.smooth_bias = 0.0
            self.smooth_bias = alpha * inst_bias + (1.0 - alpha) * self.smooth_bias
        else:
            self.smooth_bias = 0.0

        # On utilise maintenant self.smooth_bias pour toute la suite de la logique
        bias_to_use = self.smooth_bias

        # 3. GESTION DES ÉTATS
        if self.state == "EVASION":
            drives = np.array([-0.5, -0.5])
            if self.evasion_step < self.EVATION_MAX_STEP:
                self.evasion_step += 1
            else:
                self.state = self.previous_state
                self.evasion_step = 0

        elif self.state == "ALIGN_WITH_FOOD":
            drives = np.zeros(2)
            # Utilisation du biais lissé pour éviter d'osciller sur place
            if abs(bias_to_use) < 0.05:
                print("Aligné avec succès ! Passage en GO_TO_FOOD")
                self.state = "GO_TO_FOOD"
            else:
                if bias_to_use > 0:  
                    drives[0] = -0.5
                    drives[1] = 0.5
                else:         
                    drives[0] = 0.5
                    drives[1] = -0.5

        elif self.state == "GO_TO_FOOD":
            W = len(self.obstacle_cost)
            
            # x_target utilise maintenant la version filtrée et ultra stable
            x_target = (W / 2) - bias_to_use * (W / 2)
            x_target = np.clip(x_target, 0, W - 1)
            
            max_dist = max(x_target, W - 1 - x_target)
            if max_dist == 0: 
                max_dist = 1.0

            # Parabole stabilisée
            odor_cost = ((np.arange(W) - x_target) / max_dist) ** 2

            poids_obstacle = 2.5
            poids_odeur = 1.0
            total_cost = (poids_obstacle * self.obstacle_cost) + (poids_odeur * odor_cost)

            best_column = np.argmin(total_cost)
            error = (best_column - W / 2) / (W / 2)

            base_speed = 1.3
            steering_gain = 3.0
            
            drives = np.ones(2) * base_speed
            drives[0] += error * steering_gain
            drives[1] -= error * steering_gain
            drives = np.clip(drives, 0.1, 2.0)

        # Envoi des commandes
        forces = sim.get_external_force(sim.fly.name, subtract_adhesion_force=True)
        joint_angles, adhesion = self.turning_controller.step(drives)

        self.step_count += 1
        
        if self.state != "GO_TO_FOOD" or 'total_cost' not in locals():
            total_cost = np.zeros(sim.get_raw_vision(sim.fly.name)[0].shape[1] * 2)

        return joint_angles, adhesion, self.contours, self.best_contour, self.dragonfly_contours, self.state, total_cost