import numpy as np

from topoinv import horn_gradients


def scene(n, amp=0.4):
    c = 4*np.pi/n
    x, y = np.meshgrid(np.arange(n)*c, np.arange(n)*c)
    z = (np.sin(x)*np.cos(y) + .5*np.sin(2*x + 1)*np.cos(3*y - 2))*amp
    return z.astype(np.float64), c


def block_mean(a, k):
    return a.reshape(a.shape[0]//k, k, a.shape[1]//k, k).mean((1, 3))


def plane(facing, n=9, slope=1.0):
    rows, cols = np.indices((n, n), dtype=np.float64)
    east, north = cols*slope, -rows*slope
    az = np.radians({'N': 0.0, 'E': 90.0, 'S': 180.0, 'W': 270.0}[facing])
    return -(east*np.sin(az) + north*np.cos(az))


def slope_degrees(z, cellsize):
    p, q = horn_gradients(z, cellsize)
    return np.degrees(np.arctan(np.hypot(p, q)))


def corr(a, b):
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])


def rmse(a, b):
    return float(np.sqrt(((a - b)**2).mean()))
