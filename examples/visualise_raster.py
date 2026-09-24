import rasterio
import numpy as np
import matplotlib.pyplot as plt

def visualize_raster(file_path):
    with rasterio.open(file_path) as src:
        raster_data = src.read(1)
        plt.imshow(raster_data, cmap='gray')
        plt.colorbar()
        plt.title(f'Raster Visualization: {file_path}')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.show()

visualize_raster('synthetic_raster.tif')
visualize_raster('reflectance_raster.tif')