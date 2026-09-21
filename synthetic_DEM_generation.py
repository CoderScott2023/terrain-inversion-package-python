import numpy as np
import rasterio
from rasterio.transform import from_origin

width, height = 100, 100
cellsize = 4*np.pi/width

x, y = np.meshgrid(np.arange(width)*cellsize, np.arange(height)*cellsize)
z = np.sin(x)*np.cos(y) + .5*np.sin(2*x + 1)*np.cos(3*y - 2) + .25*np.sin(5*x)*np.sin(5*y)

z = z.astype(np.float32)

spatial_transform = from_origin(west=-120.0, north=45.0, xsize=cellsize, ysize=cellsize)

with rasterio.open(
    'synthetic_raster.tif',
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
