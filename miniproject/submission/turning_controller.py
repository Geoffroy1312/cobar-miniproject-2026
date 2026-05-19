import numpy as np

from flygym.examples.locomotion.cpg_network import CPGNetwork
from flygym.examples.locomotion.preprogrammed_steps import PreprogrammedSteps


_tripod_phase_biases = np.pi * np.array(
    [
        [0, 1, 0, 1, 0, 1],
        [1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1],
        [1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1],
        [1, 0, 1, 0, 1, 0],
    ]
)
_tripod_coupling_weights = (_tripod_phase_biases > 0) * 10


_tetrapod_phase_biases = np.array(
    [
        [0, 1, 2, 2, 0, 1],
        [2, 0, 1, 1, 2, 0],
        [1, 2, 0, 0, 1, 2],
        [1, 2, 0, 0, 1, 2],
        [0, 1, 2, 2, 0, 1],
        [2, 0, 1, 1, 2, 0],
    ]
) * (2 * np.pi / 3)



class TurningController:
    """Controller that uses a CPG network to generate turning behavior.

    The controller receives a 2D descending signal to modulate the
    amplitudes and frequencies of the CPGs, producing asymmetric stepping
    for turning.

    Parameters
    ----------
    timestep : float
        Simulation timestep in seconds.
    intrinsic_freqs : np.ndarray, optional
        Intrinsic frequencies of the CPGs (Hz). Shape (6,).
    intrinsic_amps : np.ndarray, optional
        Intrinsic amplitudes of the CPGs. Shape (6,).
    phase_biases : np.ndarray, optional
        Phase biases between CPGs. Shape (6, 6).
    coupling_weights : np.ndarray, optional
        Coupling weights between CPGs. Shape (6, 6).
    convergence_coefs : np.ndarray, optional
        Rate of convergence to intrinsic amplitudes. Shape (6,).
    init_phases : np.ndarray, optional
        Initial phases. Shape (6,).
    init_magnitudes : np.ndarray, optional
        Initial magnitudes. Shape (6,).
    seed : int, optional
        Random seed for CPG network initialization.
    """

    def __init__(
        self,
        timestep,
        intrinsic_freqs=np.ones(6) * 12,
        intrinsic_amps=np.ones(6) * 1,
        phase_biases=_tetrapod_phase_biases,
        coupling_weights=_tripod_coupling_weights,
        convergence_coefs=np.ones(6) * 20,
        init_phases=None,
        init_magnitudes=None,
        seed=0,
    ):
        self.preprogrammed_steps = PreprogrammedSteps()
        self.intrinsic_freqs = intrinsic_freqs
        self.cpg_network = CPGNetwork(
            timestep=timestep,
            intrinsic_freqs=intrinsic_freqs,
            intrinsic_amps=intrinsic_amps,
            coupling_weights=coupling_weights,
            phase_biases=phase_biases,
            convergence_coefs=convergence_coefs,
            init_phases=init_phases,
            init_magnitudes=init_magnitudes,
            seed=seed,
        )

    def reset(self, init_phases=None, init_magnitudes=None):
        self.cpg_network.reset(init_phases=init_phases, init_magnitudes=init_magnitudes)
        
    def step(self, action, forces, contact_threshold=1.0):
        """Step the controller forward one timestep with sensory feedback.

        Parameters
        ----------
        action : np.ndarray
            Array of shape (2,) containing the descending signal
            [delta_L, delta_R] for turning.
        forces : np.ndarray
            The external forces array passed from the simulation.
            Expected shape: (6,) for magnitudes, or (6, 3) for vectors.
        contact_threshold : float
            Force magnitude threshold to register a valid ground contact.
            
        Returns
        -------
        joint_angles : np.ndarray
        adhesion : np.ndarray
        """
        # 1. Update CPG parameters based on descending signal
        amps = np.repeat(np.abs(action[:, np.newaxis]), 3, axis=1).ravel()
        freqs = self.intrinsic_freqs.copy()
        freqs[:3] *= 1 if action[0] > 0 else -1
        freqs[3:] *= 1 if action[1] > 0 else -1
        self.cpg_network.intrinsic_amps = amps
        self.cpg_network.intrinsic_freqs = freqs

        # 2. Step the CPG network forward normally
        self.cpg_network.step()

        # 3. Apply Sensory Feedback (Phase Resetting)
        # Calculate force magnitudes. 
        # If 'forces' contains 3D vectors (x,y,z), we take the norm. 
        # If it's already 1D, we take the absolute value.
        if forces.ndim == 2:
            force_mags = np.linalg.norm(forces, axis=1)
        else:
            force_mags = np.abs(forces)
            
        for i, leg in enumerate(self.preprogrammed_steps.legs):
            current_phase_mod = self.cpg_network.curr_phases[i] % (2 * np.pi)
            
            # THE FIX: Only listen for touchdown in the last 25% of the cycle 
            # (when the leg is definitely swinging downward)
            in_swing_descent = (1.5 * np.pi) < current_phase_mod < (1.8 * np.pi)
            
            if force_mags[i] > contact_threshold and in_swing_descent:
                # DEBUG PRINT: This will tell us exactly what is triggering the reset
                
                
                phase_correction = (2 * np.pi) - current_phase_mod
                self.cpg_network.curr_phases[i] += phase_correction

        # 4. Generate joint angles and adhesion from the (potentially modified) phases
        joint_angles = np.zeros((6, 7))
        adhesion = np.zeros(6)
        
        for i, leg in enumerate(self.preprogrammed_steps.legs):
            joint_angles[i] = self.preprogrammed_steps.get_joint_angles(
                leg,
                self.cpg_network.curr_phases[i],
                self.cpg_network.curr_magnitudes[i],
            )
            adhesion[i] = self.preprogrammed_steps.get_adhesion_onoff(
                leg, self.cpg_network.curr_phases[i]
            )
        
        return joint_angles.ravel(), adhesion