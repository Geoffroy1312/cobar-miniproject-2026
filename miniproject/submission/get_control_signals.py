import numpy as np

def contour_to_aversive_odor(best_contour, olfaction, grass_in_middle, direction):
    fake_odor = np.zeros((4, 1))

    if best_contour is not None:
        
        distance = best_contour[1]
        size = best_contour[2]

        attractive_intensities = np.average(
            olfaction.reshape(2, 2), axis=0, weights=[9, 1]
        )

        #check if a grass is present in the middle of the path, force an evasive maneuver to one side
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
        #other type of grass
        else:
            if distance < 0:
                #something in the left side, stimulate right side
                fake_odor[0, 0] += size
            else:
                #something on the left side, stimulate left side
                fake_odor[1, 0] += size

    return fake_odor, direction


def odor_intensity_to_control_signal(
    odor_intensities,
    state,
    aversive_gain,
    attractive_gain=-30
):
    """Convert odor sensor readings to a turning control signal."""

    attractive_intensities = np.average(
        odor_intensities[:, 0].reshape(2, 2), axis=0, weights=[9, 1]
    )
    
    #odor bias
    attractive_bias = (
            attractive_gain
            * (attractive_intensities[0] - attractive_intensities[1])
            / attractive_intensities.mean()
            if attractive_intensities.mean() != 0
            else 0
        )
    
    #vision bias, check if it exists
    if odor_intensities.shape[1] > 1:
        aversive_intensities = np.average(
            odor_intensities[:, 1].reshape(2, 2), axis=0, weights=[10, 0]
        )
        
        aversive_bias = (
            aversive_gain * (aversive_intensities[0] - aversive_intensities[1]) / 512
        )

    else:
        aversive_bias = 0

    #clip the odor bias to avoid saturating the drives
    clipped_attractive_bias = np.clip(attractive_bias, -1, 1)
    effective_bias = aversive_bias + clipped_attractive_bias
    effective_bias_norm = np.tanh(effective_bias**2) * np.sign(effective_bias)

    #make the fly rotate to go toward food
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

    #activate vision bias 
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

    #go faster to avoid the dragonfly
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

    return control_signal, aversive_bias, clipped_attractive_bias
