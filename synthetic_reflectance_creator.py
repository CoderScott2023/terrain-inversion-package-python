import rasterio
import numpy as np

NODATA = 0
LIT = 1
SELF_SHADOWED = 2


def read_raster(file_path):
    with rasterio.open(file_path) as src:
        topo = src.read(1)
        return topo


def horn_gradients(topo, cellsize=1.0):
    p = np.pad(np.asarray(topo, dtype=np.float64), 1, mode='edge')
    a, b, c = p[:-2, :-2], p[:-2, 1:-1], p[:-2, 2:]
    d, f = p[1:-1, :-2], p[1:-1, 2:]
    g, h, i = p[2:, :-2], p[2:, 1:-1], p[2:, 2:]
    dzdx = ((c + 2.0*f + i) - (a + 2.0*d + g)) / (8.0*cellsize)
    dzdy = ((g + 2.0*h + i) - (a + 2.0*b + c)) / (8.0*cellsize)
    return dzdx, dzdy


def surface_normals(topo, cellsize=1.0):
    dzdx, dzdy = horn_gradients(topo, cellsize)
    norm = np.sqrt(dzdx**2 + dzdy**2 + 1.0)
    return np.stack((-dzdx/norm, dzdy/norm, 1.0/norm), axis=-1)


def direction_vector(azimuth, altitude):
    return np.array([np.sin(azimuth)*np.cos(altitude),
                     np.cos(azimuth)*np.cos(altitude),
                     np.sin(altitude)])


def lambert(mu0, mu, phase, albedo=1.0):
    return albedo * mu0


def lommel_seeliger(mu0, mu, phase, albedo=1.0):
    return albedo * 2.0 * mu0 / np.where(mu0 + mu > 0.0, mu0 + mu, np.inf)


def mcewen_L(phase):
    g = np.degrees(phase)
    return np.clip(1.0 - 0.019*g + 2.42e-4*g**2 - 1.46e-6*g**3, 0.0, 1.0)


def lunar_lambert(mu0, mu, phase, albedo=1.0, L=None):
    if L is None:
        L = mcewen_L(phase)
    return albedo * (L * lommel_seeliger(mu0, mu, phase) + (1.0 - L) * mu0)


MODELS = {
    'lambert': lambert,
    'lommel_seeliger': lommel_seeliger,
    'lunar_lambert': lunar_lambert,
}


def create_reflectance_raster(topo, azimuth, altitude=None, zenith=None, cellsize=1.0,
                              view_azimuth=0.0, view_altitude=90.0, albedo=1.0,
                              model='lunar_lambert', rad=False, return_mask=False):
    if zenith is None:
        if altitude is None:
            raise ValueError("Either altitude or zenith must be provided.")
        zenith = (np.pi/2 - altitude) if rad else (90 - altitude)

    if not rad:
        zenith = np.radians(zenith)
        azimuth = np.radians(azimuth)
        view_azimuth = np.radians(view_azimuth)
        view_altitude = np.radians(view_altitude)

    model_fn = model if callable(model) else MODELS.get(model)
    if model_fn is None:
        raise ValueError(f"unknown model {model!r}; choose from {sorted(MODELS)}")

    topo = np.asarray(topo, dtype=np.float64)
    normals = surface_normals(topo, cellsize)
    s_vec = direction_vector(azimuth, np.pi/2 - zenith)
    v_vec = direction_vector(view_azimuth, view_altitude)

    mu0 = normals @ s_vec
    mu = normals @ v_vec
    phase = float(np.arccos(np.clip(s_vec @ v_vec, -1.0, 1.0)))

    bad = ~np.isfinite(topo) | ~np.isfinite(mu0) | ~np.isfinite(mu)
    lit = (mu0 > 0.0) & (mu > 0.0) & ~bad
    mask = np.where(bad, NODATA, np.where(lit, LIT, SELF_SHADOWED)).astype(np.uint8)

    values = model_fn(np.where(lit, mu0, 0.0), np.where(lit, mu, 1.0), phase, albedo)
    reflectance = np.where(lit, np.broadcast_to(values, topo.shape), 0.0)
    reflectance = np.clip(reflectance, 0.0, None).astype(np.float32)

    return (reflectance, mask) if return_mask else reflectance


def write_raster(src, reflectance, output_file):
    with rasterio.open(
        output_file,
        'w',
        driver='GTiff',
        height=reflectance.shape[0],
        width=reflectance.shape[1],
        count=1,
        dtype=reflectance.dtype,
        crs=src.crs,
        transform=src.transform,
    ) as dst:
        dst.write(reflectance, 1)


if __name__ == '__main__':
    input_path = 'synthetic_raster.tif'
    output_path = 'reflectance_raster.tif'
    mask_path = 'reflectance_mask.tif'

    topo_data = read_raster(input_path)

    with rasterio.open(input_path) as src_file:
        reflectance_data, mask_data = create_reflectance_raster(
            topo_data, azimuth=45, altitude=30, cellsize=src_file.transform.a,
            model='lunar_lambert', return_mask=True)
        write_raster(src_file, reflectance_data, output_path)
        write_raster(src_file, mask_data, mask_path)
