import cv2
import numpy as np
import imageio
from IPython.display import Video, display
from matplotlib import pyplot as plt
from tqdm import trange
from flygym.compose import ActuatorType
from miniproject.simulation import MiniprojectSimulation

# Import du contrôleur de marche pour transformer les drives en angles
from submission.turning_controller import TurningController
# Import de votre nouveau modèle de vision
from submission.vision_model import FlyVisionModel

# 1. Initialisation
sim = MiniprojectSimulation(level=3, seed=1)
turning_controller = TurningController(sim.timestep)
vision_model = FlyVisionModel()

# 2. Initialiser le VideoWriter
video_filename = "fly_vision_occupancy.mp4"
writer = imageio.get_writer(video_filename, fps=30) 

# Boucle de simulation
for step in trange(50000):
    # Drives constants comme demandé
    drives = np.array([1.0, 1.0])
    
    # Conversion en commandes pour la simulation
    joint_angles, adhesion = turning_controller.step(drives)
    
    sim.set_actuator_inputs(sim.fly.name, ActuatorType.POSITION, joint_angles)
    sim.set_actuator_inputs(sim.fly.name, ActuatorType.ADHESION, adhesion)
    sim.step()
    sim.render_as_needed()
    
    # 3. Capturer la vision et créer la vidéo (tous les 20 pas)
    if step % 20 == 0:
        raw_vision = sim.get_raw_vision(sim.fly.name)
        
        left_eye = raw_vision[0]
        right_eye = raw_vision[1]
        
        # Concaténer la vision des deux yeux
        combined_vision = np.hstack((left_eye, right_eye)).astype(np.uint8)
        
        # Obtenir la grille d'occupation 1D
        occupancy_1d = vision_model.get_1d_occupancy_grid(combined_vision)
        
        # 4. Créer une représentation visuelle du tableau d'occupation
        height, width, _ = combined_vision.shape
        occupancy_height = 40  # Hauteur de la barre d'occupation en pixels
        occupancy_img = np.zeros((occupancy_height, width, 3), dtype=np.uint8)
        
        # Colorier la barre : Rouge pour occupé (1), Blanc pour libre (0)
        for i in range(width):
            if occupancy_1d[i] == 1:
                occupancy_img[:, i] = [255, 0, 0]    # Rouge
            else:
                occupancy_img[:, i] = [255, 255, 255]  # Blanc
                
        # 5. Coller la barre d'occupation en dessous de l'image de vision
        final_frame = np.vstack((combined_vision, occupancy_img))
        
        # Ajouter à la vidéo
        writer.append_data(final_frame)

# 6. Fermer le fichier vidéo
writer.close()

# Afficher le rendu global de la simulation (vue externe de la mouche)
sim.renderer.show_in_notebook()

# 7. Afficher la vidéo de la vision de la mouche + grille d'occupation
display(Video(video_filename, embed=True, width=800))