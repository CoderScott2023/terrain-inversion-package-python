import numpy as np
import rasterio

import src.core.synthetic as pc

pc.create_synthetic_dem('test.tif', 100, 100, 4*np.pi/100)
topo = pc.tif_to_array('reflectance_raster.tif')
reflectance, mask = pc.create_reflectance_raster(topo, azimuth=45, altitude=30, cellsize=4*np.pi/100, model='lunar_lambert', return_mask=True)
inverted = pc.invert_reflectance(reflectance, azimuth=45, altitude=30, cellsize=4*np.pi/100, model='lunar_lambert', maxiter=2000, eps=1e-6, alpha=1e-7)


import matplotlib.pyplot as plt

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(inverted, cmap='gray')
plt.title('Inverted')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(topo, cmap='gray')
plt.title('Synthetic DEM')
plt.axis('off')

plt.tight_layout()
plt.show()

print(pc.dem_error(topo, inverted))