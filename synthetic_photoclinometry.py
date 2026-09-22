import rasterio
import numpy as np

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
