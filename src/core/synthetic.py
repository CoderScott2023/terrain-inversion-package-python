import numpy as np
import rasterio
from rasterio.transform import from_origin

from src.photometry.BRDFs import MODELS

def create_synthetic_dem(output_path, width=100, height=100, cellsize=None):
    if cellsize is None:
        cellsize = 4*np.pi/width

    x, y = np.meshgrid(np.arange(width)*cellsize, np.arange(height)*cellsize)
    z = np.sin(x)*np.cos(y) + .5*np.sin(2*x + 1)*np.cos(3*y - 2) + .25*np.sin(5*x)*np.sin(5*y)
    z = z.astype(np.float32)

    spatial_transform = from_origin(west=-120.0, north=45.0, xsize=cellsize, ysize=cellsize)

    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype=z.dtype,
        crs='EPSG:4326',
        transform=spatial_transform,
    ) as dst:
        dst.write(z, 1)

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

def tif_to_array(tif_path):
    with rasterio.open(tif_path) as src:
        array = src.read(1)
    return array

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
from scipy.optimize import minimize

from synthetic_reflectance_creator import MODELS, direction_vector, horn_gradients, surface_normals


def estimate_valid(image, nodata=None, shadow_threshold=None, saturation_threshold=None):
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.integer):
        full = float(np.iinfo(image.dtype).max)
        low = 0.02 * full if shadow_threshold is None else shadow_threshold
        high = full - 1.0 if saturation_threshold is None else saturation_threshold
        finite = np.ones(image.shape, dtype=bool)
    else:
        low = 0.02 if shadow_threshold is None else shadow_threshold
        high = np.inf if saturation_threshold is None else saturation_threshold
        finite = np.isfinite(image)

    valid = finite & (image > low) & (image < high)
    if nodata is not None:
        valid &= image != nodata
    return valid


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

    model_fn = model if callable(model) else MODELS.get(model)
    if model_fn is None:
        raise ValueError(f"unknown model {model!r}; choose from {sorted(MODELS)}")

    s_vec = direction_vector(azimuth, np.pi/2 - zenith)
    v_vec = direction_vector(view_azimuth, view_altitude)
    phase = float(np.arccos(np.clip(s_vec @ v_vec, -1.0, 1.0)))
    return s_vec, v_vec, phase, model_fn


_KX = {(-1, 1): 1.0, (0, 1): 2.0, (1, 1): 1.0, (-1, -1): -1.0, (0, -1): -2.0, (1, -1): -1.0}
_KY = {(1, -1): 1.0, (1, 0): 2.0, (1, 1): 1.0, (-1, -1): -1.0, (-1, 0): -2.0, (-1, 1): -1.0}


def horn_adjoint(Wx, Wy, cellsize=1.0):
    rows, cols = Wx.shape
    out = np.zeros((rows + 2, cols + 2), dtype=np.float64)
    for kernel, W in ((_KX, Wx), (_KY, Wy)):
        for (dr, dc), c in kernel.items():
            out[1+dr:1+dr+rows, 1+dc:1+dc+cols] += (c / (8.0*cellsize)) * W
    out[1, :] += out[0, :]
    out[-2, :] += out[-1, :]
    out[:, 1] += out[:, 0]
    out[:, -2] += out[:, -1]
    return out[1:-1, 1:-1]


def gradient_adjoint(W, axis):
    out = np.zeros_like(W)
    Wm = np.moveaxis(W, axis, 0)
    om = np.moveaxis(out, axis, 0)
    om[1] += Wm[0]
    om[0] -= Wm[0]
    om[2:] += Wm[1:-1] / 2.0
    om[:-2] -= Wm[1:-1] / 2.0
    om[-1] += Wm[-1]
    om[-2] -= Wm[-1]
    return out


def make_objective(reflectance, valid, s_vec, v_vec, phase, model_fn,
                   cellsize=1.0, albedo=1.0, alpha=1e-7, eps=1e-6):
    rows, cols = reflectance.shape
    h = cellsize
    n_valid = max(int(valid.sum()), 1)
    n_pix = rows * cols

    def objective_function(z_flat):
        Z = z_flat.reshape((rows, cols))
        p, q = horn_gradients(Z, h)
        N = np.sqrt(p**2 + q**2 + 1.0)
        mu0 = (-p*s_vec[0] + q*s_vec[1] + s_vec[2]) / N
        mu = (-p*v_vec[0] + q*v_vec[1] + v_vec[2]) / N
        lit = (mu0 > 0.0) & (mu > 0.0)

        m0 = np.where(lit, mu0, 0.0)
        m1 = np.where(lit, mu, 1.0)
        rendered = np.where(lit, model_fn(m0, m1, phase, albedo), 0.0)

        resid = np.where(valid, rendered - reflectance, 0.0)
        data_loss = np.sum(resid**2) / n_valid

        f_mu0 = (model_fn(m0 + eps, m1, phase, albedo)
                 - model_fn(m0 - eps, m1, phase, albedo)) / (2.0*eps)
        f_mu = (model_fn(m0, m1 + eps, phase, albedo)
                - model_fn(m0, m1 - eps, phase, albedo)) / (2.0*eps)
        f_mu0 = np.where(lit, f_mu0, 0.0)
        f_mu = np.where(lit, f_mu, 0.0)

        inv_N2 = 1.0 / N**2
        dmu0_dp = -s_vec[0]/N - mu0*p*inv_N2
        dmu0_dq = s_vec[1]/N - mu0*q*inv_N2
        dmu_dp = -v_vec[0]/N - mu*p*inv_N2
        dmu_dq = v_vec[1]/N - mu*q*inv_N2

        g = 2.0 * resid / n_valid
        grad = horn_adjoint(g*(f_mu0*dmu0_dp + f_mu*dmu_dp),
                            g*(f_mu0*dmu0_dq + f_mu*dmu_dq), h)

        d2Z_dx2 = np.gradient(p, axis=1) / h
        d2Z_dy2 = np.gradient(q, axis=0) / h
        smoothness_loss = np.sum(d2Z_dx2**2 + d2Z_dy2**2) / n_pix
        grad += horn_adjoint(gradient_adjoint(2.0*alpha*d2Z_dx2/(h*n_pix), 1),
                             gradient_adjoint(2.0*alpha*d2Z_dy2/(h*n_pix), 0), h)

        return data_loss + alpha*smoothness_loss, grad.ravel()

    return objective_function


def invert_reflectance(reflectance, azimuth, altitude=None, zenith=None, cellsize=1.0,
                       view_azimuth=0.0, view_altitude=90.0, albedo=1.0,
                       model='lunar_lambert', rad=False, valid=None, alpha=1e-7,
                       maxiter=2000, eps=1e-6, z0=None, ftol=1e-16, gtol=1e-12):
    reflectance = np.asarray(reflectance, dtype=np.float64)
    if valid is None:
        valid = estimate_valid(reflectance)

    s_vec, v_vec, phase, model_fn = prepare_geometry(
        azimuth, altitude, zenith, view_azimuth, view_altitude, model, rad)

    rows, cols = reflectance.shape
    objective_function = make_objective(reflectance, valid, s_vec, v_vec, phase, model_fn,
                                        cellsize, albedo, alpha, eps)

    x0 = np.zeros(rows*cols, dtype=np.float64) if z0 is None else np.asarray(z0, dtype=np.float64).ravel()
    res = minimize(objective_function, x0, method='L-BFGS-B', jac=True,
                   options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol})

    final_dem = res.x.reshape((rows, cols))
    return final_dem - final_dem.mean()


def invert_reflectance_tif(reflectance_tif, output_tif, azimuth, cellsize=None, **kwargs):
    with rasterio.open(reflectance_tif) as src:
        reflectance = src.read(1)
        profile = src.profile
        if cellsize is None:
            cellsize = src.transform.a

    dem = invert_reflectance(reflectance, azimuth, cellsize=cellsize, **kwargs)

    profile.update(driver='GTiff', count=1, dtype='float32')
    with rasterio.open(output_tif, 'w', **profile) as dst:
        dst.write(dem.astype(np.float32), 1)
    return dem

def dem_error(dem1, dem2, valid=None):
    dem1 = np.asarray(dem1, dtype=np.float64)
    dem2 = np.asarray(dem2, dtype=np.float64)
    if valid is None:
        valid = np.isfinite(dem1) & np.isfinite(dem2)
    diff = dem1 - dem2
    diff[~valid] = 0.0
    n_valid = max(int(valid.sum()), 1)
    rmse = np.sqrt(np.sum(diff**2) / n_valid)
    mae = np.sum(np.abs(diff)) / n_valid
    return rmse, mae