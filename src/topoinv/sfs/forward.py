import numpy as np

from ..core.masks import LIT, NODATA, SELF_SHADOWED
from ..geometry.directions import direction_vector
from ..geometry.grid import surface_normals
from ..photometry.brdf import resolve_model


def prepare_geometry(azimuth, altitude=None, zenith=None, view_azimuth=0.0,
                     view_altitude=90.0, model='lunar_lambert', rad=False):
    if zenith is None:
        if altitude is None:
            raise ValueError("Either altitude or zenith must be provided.")
        zenith = (np.pi/2 - altitude) if rad else (90 - altitude)

    if not rad:
        zenith = np.radians(zenith)
        azimuth = np.radians(azimuth)
        view_azimuth = np.radians(view_azimuth)
        view_altitude = np.radians(view_altitude)

    model_fn = resolve_model(model)
    s_vec = direction_vector(azimuth, np.pi/2 - zenith)
    v_vec = direction_vector(view_azimuth, view_altitude)
    phase = float(np.arccos(np.clip(s_vec @ v_vec, -1.0, 1.0)))
    return s_vec, v_vec, phase, model_fn


def render(topo, azimuth, altitude=None, zenith=None, cellsize=1.0,
           view_azimuth=0.0, view_altitude=90.0, albedo=1.0,
           model='lunar_lambert', rad=False, return_mask=False):
    s_vec, v_vec, phase, model_fn = prepare_geometry(
        azimuth, altitude, zenith, view_azimuth, view_altitude, model, rad)

    topo = np.asarray(topo, dtype=np.float64)
    normals = surface_normals(topo, cellsize)
    mu0 = normals @ s_vec
    mu = normals @ v_vec

    bad = ~np.isfinite(topo) | ~np.isfinite(mu0) | ~np.isfinite(mu)
    lit = (mu0 > 0.0) & (mu > 0.0) & ~bad
    mask = np.where(bad, NODATA, np.where(lit, LIT, SELF_SHADOWED)).astype(np.uint8)

    values = model_fn(np.where(lit, mu0, 0.0), np.where(lit, mu, 1.0), phase, albedo)
    reflectance = np.where(lit, np.broadcast_to(values, topo.shape), 0.0)
    reflectance = np.clip(reflectance, 0.0, None).astype(np.float32)

    return (reflectance, mask) if return_mask else reflectance
