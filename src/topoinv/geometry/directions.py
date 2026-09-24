import numpy as np


def direction_vector(azimuth, altitude):
    return np.array([np.sin(azimuth)*np.cos(altitude),
                     np.cos(azimuth)*np.cos(altitude),
                     np.sin(altitude)])
