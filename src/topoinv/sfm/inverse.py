import numpy as np
import rasterio

def invert_reflectance_sfm(
    reflectance,              # list/tuple of 2D reflectance images
    azimuth,                  # sun azimuth: scalar or one per image
    altitude=None,            # sun altitude: scalar or one per image
    zenith=None,              # alternatively specify sun zenith
    cellsize=1.0,

    view_azimuth=0.0,         # camera azimuth: scalar or one per image
    view_altitude=90.0,       # camera elevation: scalar or one per image

    rad=False,
    valid=None,
    alpha=1e-7,

    maxiter=10000,
    eps=1e-6,

    z0=None,

    ftol=1e-12,
    gtol=1e-10,

    scale=None,
    return_result=False,
    warn_unconverged=True,
):

    

    return