import warnings

import numpy as np
from scipy.optimize import minimize

from ..core.masks import estimate_valid
from ..geometry.grid import gradient_adjoint, horn_adjoint, horn_gradients
from .forward import prepare_geometry


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


PLAUSIBLE_PEAK = 2.0


def to_reflectance(image, scale=None):
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.integer) and scale is None:
        raise ValueError(
            f"image dtype {image.dtype} is digital number, not reflectance. Photometric "
            "models return I/F in [0, 1], so raw DN gives meaningless residuals. Pass "
            "scale=1/{max} to rescale, or radiometrically calibrate to I/F first "
            "(for LROC NAC: lronaccal).".format(max=np.iinfo(image.dtype).max))
    out = np.asarray(image, dtype=np.float64)
    if scale is not None:
        out = out * scale
    finite = out[np.isfinite(out)]
    if finite.size and finite.max() > PLAUSIBLE_PEAK:
        warnings.warn(
            f"peak value {finite.max():.4g} exceeds plausible reflectance "
            f"({PLAUSIBLE_PEAK}); check radiometric scaling",
            RuntimeWarning, stacklevel=3)
    return out


def invert(reflectance, azimuth, altitude=None, zenith=None, cellsize=1.0,
           view_azimuth=0.0, view_altitude=90.0, albedo=1.0,
           model='lunar_lambert', rad=False, valid=None, alpha=1e-7,
           maxiter=10000, eps=1e-6, z0=None, ftol=1e-12, gtol=1e-10,
           scale=None, return_result=False, warn_unconverged=True):
    reflectance = to_reflectance(reflectance, scale)
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

    if warn_unconverged and not res.success:
        warnings.warn(f"optimiser did not converge after {res.nit} iterations: "
                      f"{res.message}", RuntimeWarning, stacklevel=2)

    final_dem = res.x.reshape((rows, cols))
    dem = final_dem - final_dem.mean()
    return (dem, res) if return_result else dem


def invert_multilook(reflectances, azimuths, altitudes=None, zeniths=None, cellsizes=1.0,
                     view_azimuths=0.0, view_altitudes=90.0, albedo=1.0,
                     model='lunar_lambert', rad=False, valid=None, alpha=1e-7,
                     maxiter=10000, eps=1e-6, z0=None, ftol=1e-12, gtol=1e-10,
                     scale=None, return_result=False, warn_unconverged=True):

    objectives = []

    for i, each_reflectance in enumerate(reflectances):

        each_reflectance = to_reflectance(each_reflectance, scale)

        if valid is None:
            each_valid = estimate_valid(each_reflectance)
        else:
            each_valid = valid[i]

        each_s_vec, each_v_vec, each_phase, model_fn = prepare_geometry(
            azimuths[i],
            altitudes[i] if altitudes is not None else None,
            zeniths[i] if zeniths is not None else None,
            view_azimuths[i] if view_azimuths is not None else None,
            view_altitudes[i] if view_altitudes is not None else None,
            model,
            rad
        )

        each_cellsize = cellsizes[i] if np.ndim(cellsizes) > 0 else cellsizes

        objectives.append(
            make_objective(
                each_reflectance,
                each_valid,
                each_s_vec,
                each_v_vec,
                each_phase,
                model_fn,
                each_cellsize,
                albedo,
                alpha,
                eps
            )
        )

    def objective_function(z_flat):
        total_loss = 0.0
        total_grad = np.zeros_like(z_flat)

        for objective in objectives:
            loss, grad = objective(z_flat)
            total_loss += loss
            total_grad += grad

        return total_loss, total_grad

    rows, cols = reflectances[0].shape

    x0 = (
        np.zeros(rows * cols, dtype=np.float64)
        if z0 is None
        else np.asarray(z0, dtype=np.float64).ravel()
    )

    res = minimize(
        objective_function,
        x0,
        method='L-BFGS-B',
        jac=True,
        options={
            'maxiter': maxiter,
            'ftol': ftol,
            'gtol': gtol
        }
    )

    if warn_unconverged and not res.success:
        warnings.warn(
            f"optimiser did not converge after {res.nit} iterations: "
            f"{res.message}",
            RuntimeWarning,
            stacklevel=2
        )

    final_dem = res.x.reshape((rows, cols))
    dem = final_dem - final_dem.mean()

    return (dem, res) if return_result else dem


        


    


def invert_file(reflectance_tif, output_tif, azimuth, cellsize=None, scale=None, **kwargs):
    from ..core.io import read_raster_with_grid, write_array

    reflectance, grid_size, profile = read_raster_with_grid(reflectance_tif)
    if cellsize is None:
        cellsize = grid_size

    if kwargs.get('valid') is None:
        nodata = profile.get('nodata')
        kwargs['valid'] = estimate_valid(
            to_reflectance(reflectance, scale),
            nodata=None if nodata is None else nodata * (1.0 if scale is None else scale))

    dem = invert(reflectance, azimuth, cellsize=cellsize, scale=scale, **kwargs)

    profile = dict(profile)
    profile.pop('nodata', None)
    write_array(output_tif, dem.astype(np.float32), profile=profile)
    return dem
