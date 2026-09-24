import numpy as np
import pytest

from helpers import scene
from topoinv import gradient_adjoint, horn_adjoint
from topoinv import horn_gradients


@pytest.mark.parametrize('n,cell', [(7, 1.0), (12, 0.35), (16, 2.5)])
def test_horn_operator_adjoint_identity(n, cell):
    rng = np.random.default_rng(n)
    z = rng.normal(size=(n, n))
    wx, wy = rng.normal(size=(n, n)), rng.normal(size=(n, n))
    px, py = horn_gradients(z, cell)
    lhs = np.sum(px*wx) + np.sum(py*wy)
    rhs = np.sum(z*horn_adjoint(wx, wy, cell))
    assert abs(lhs - rhs) <= 1e-10*max(abs(lhs), 1.0)


@pytest.mark.parametrize('axis', [0, 1])
@pytest.mark.parametrize('n', [6, 11])
def test_gradient_operator_adjoint_identity(axis, n):
    rng = np.random.default_rng(n + axis)
    z, w = rng.normal(size=(n, n)), rng.normal(size=(n, n))
    lhs = np.sum(np.gradient(z, axis=axis)*w)
    rhs = np.sum(z*gradient_adjoint(w, axis))
    assert abs(lhs - rhs) <= 1e-10*max(abs(lhs), 1.0)


def test_horn_adjoint_conserves_a_constant_field():
    z, cell = scene(10)[0], 0.4
    wx = np.ones((10, 10))
    wy = np.zeros((10, 10))
    px, _ = horn_gradients(z, cell)
    assert np.isclose(np.sum(px*wx), np.sum(z*horn_adjoint(wx, wy, cell)), atol=1e-10)
