import matplotlib.pyplot as plt
import numpy as np

from topoinv import invert, render, ripple_dem, slope_degrees, summary

truth, cellsize = ripple_dem(width=100, height=100, amplitude=0.4, detail=False)
truth = truth - truth.mean()

reflectance, mask = render(truth, azimuth=45, altitude=40, cellsize=cellsize,
                           model='lunar_lambert', return_mask=True)
recovered = invert(reflectance, azimuth=45, altitude=40, cellsize=cellsize,
                   model='lunar_lambert', alpha=1e-7, maxiter=3000)

print(f'mean slope {slope_degrees(truth, cellsize).mean():.1f} deg')
for key, value in summary(recovered, truth, cellsize).items():
    print(f'{key:16s} {value:+.4f}')

fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
for ax, data, title in zip(axes, (truth, reflectance, recovered),
                           ('truth DEM', 'rendered reflectance', 'recovered DEM')):
    im = ax.imshow(data, cmap='gray')
    ax.set_title(title)
    ax.axis('off')
    fig.colorbar(im, ax=ax, fraction=0.046)
plt.tight_layout()
plt.show()
