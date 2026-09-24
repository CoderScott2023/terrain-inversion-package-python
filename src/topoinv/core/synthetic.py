import numpy as np


def ripple_dem(width=100, height=100, cellsize=None, amplitude=1.0, detail=True):
    if cellsize is None:
        cellsize = 4*np.pi/width
    x, y = np.meshgrid(np.arange(width)*cellsize, np.arange(height)*cellsize)
    z = np.sin(x)*np.cos(y) + .5*np.sin(2*x + 1)*np.cos(3*y - 2)
    if detail:
        z = z + .25*np.sin(5*x)*np.sin(5*y)
    return (z*amplitude).astype(np.float32), cellsize


def create_synthetic_dem(output_path, width=100, height=100, cellsize=None,
                         amplitude=1.0, detail=True):
    from rasterio.transform import from_origin

    from .io import write_array

    z, cellsize = ripple_dem(width, height, cellsize, amplitude, detail)
    transform = from_origin(west=-120.0, north=45.0, xsize=cellsize, ysize=cellsize)
    write_array(output_path, z, transform=transform, crs='EPSG:4326')
    return z, cellsize
