import numpy as np
import rasterio


def read_raster(file_path, band=1):
    with rasterio.open(file_path) as src:
        return src.read(band)


def read_raster_with_grid(file_path, band=1):
    with rasterio.open(file_path) as src:
        return src.read(band), src.transform.a, src.profile


def write_array(output_path, array, transform=None, crs=None, profile=None):
    if profile is None:
        profile = {'driver': 'GTiff', 'crs': crs, 'transform': transform}
    profile = dict(profile)
    profile.update(driver='GTiff', height=array.shape[0], width=array.shape[1],
                   count=1, dtype=array.dtype.name)
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(array, 1)


def write_like(reference_path, array, output_path):
    with rasterio.open(reference_path) as src:
        write_array(output_path, array, transform=src.transform, crs=src.crs)
