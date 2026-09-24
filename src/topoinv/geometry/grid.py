import numpy as np

_KX = {(-1, 1): 1.0, (0, 1): 2.0, (1, 1): 1.0, (-1, -1): -1.0, (0, -1): -2.0, (1, -1): -1.0}
_KY = {(1, -1): 1.0, (1, 0): 2.0, (1, 1): 1.0, (-1, -1): -1.0, (-1, 0): -2.0, (-1, 1): -1.0}


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
