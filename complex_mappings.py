import numpy as np
import time
from numpy.polynomial import Polynomial
import plotly.graph_objects as go

A_aff = complex(1, 0)
B_aff = complex(1, 0)
A_spi = complex(1, 0)
B_spi = complex(0, np.pi/8)


def cx_affine_trans(x, y, A, B, inv=False):
    """Returns w, where w = Az + B, and z = x + iy"""
    z = x + 1j*y
    if not inv:
        w = A*z + B
    else:
        w = (z-B)/A
    return w.real, w.imag


def cx_spiral_trans(x, y, A, B, inv=False):
    """Returns w, where w = A exp(Bi), and z = x + iy"""
    z = x + 1j*y
    if not inv:
        w = A*np.exp(B)*z
    else:
        w = np.exp(-B)*z/A
    return w.real, w.imag


def cx_any_trans(x, y, func):
    """Returns w, where w = func(z), and z = x + iy"""
    z = x + 1j * y
    w = np.array([*map(func, z)])
    return w.real, w.imag


# w = cx_any_trans(np.array([1, 2]), np.array([1, 2]), lambda z: z**2)
# print(w)


x_min = -np.pi
x_max = np.pi
y_min = -np.pi
y_max = np.pi
h = .1


def func(z):
    w = np.sin(z)
    return w


fig = go.FigureWidget()

# Horizontal lines
hor_lines = []
hor_along = np.arange(x_min, x_max+1, h)
for pos in np.arange(y_min, y_max+1, h):
    ver_pos = pos*np.ones_like(hor_along)
    # fig.add_trace(go.Scatter(x=hor_along, y=ver_pos, mode='lines', line={'color': 'white'}))
    hor_lines.append((hor_along, ver_pos))

# Vertical lines
ver_lines = []
ver_along = np.arange(y_min, y_max+1, h)
for pos in np.arange(x_min, x_max+1, h):
    hor_pos = pos*np.ones_like(ver_along)
    # fig.add_trace(go.Scatter(x=hor_pos, y=ver_along, mode='lines', line={'color': 'white'}))
    ver_lines.append((ver_along, hor_pos))

# Transformed Horizontal lines
for hor_along, ver_pos in hor_lines:
    x_trans, y_trans = cx_any_trans(hor_along, ver_pos, func)
    fig.add_trace(go.Scattergl(x=x_trans, y=y_trans, mode='lines', line={'color': 'white'}))

# Transformed Vertical lines
for ver_along, hor_pos in hor_lines:
    x_trans, y_trans = cx_any_trans(hor_along, ver_pos, func)
    fig.add_trace(go.Scattergl(x=hor_pos, y=ver_along, mode='lines', line={'color': 'white'}))

fig.update_layout(template='plotly_dark', showlegend=False)
fig.update_yaxes(scaleanchor="x", scaleratio=1)
fig.show()