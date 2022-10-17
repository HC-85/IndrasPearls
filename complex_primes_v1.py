import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Plots the complex primes(*+-(1|i)) inside an annulus given by inner radius (inclusive) and outer radius (exclusive)


def cx_tup_mult(a, b):
    res = complex(a[0], a[1])*complex(b[0], b[1])
    return res.real, res.imag


def iterable_to_lists(iterable, pos):
    newlist = []
    for a in iterable:
        newlist.append(a[pos])
    return newlist


def tup_90_rot(tup):
    return -tup[1], tup[0]


def perms_in_inter(inner_radius, outer_radius):
    perms = [(a, b) for a in range(outer_radius + 1) for b in range(outer_radius + 1)]
    perms_abs = []
    for a, b in perms:
        perms_abs.append(np.sqrt(a ** 2 + b ** 2))
    in_range = [(a >= inner_radius) & (a < outer_radius) for a in perms_abs]
    perms_in_depth = list(pd.Series(perms)[in_range])
    return perms_in_depth


inner_radius = 0
outer_radius = 20
factors = perms_in_inter(0, outer_radius)
factors.remove((1, 0))
factors.remove((0, 1))
for _ in range(4):
    for tupol in factors[:]:
        factors.append(tup_90_rot(tupol))
perms_in_depth = perms_in_inter(inner_radius, outer_radius)
products = [cx_tup_mult(a, b) for a in factors for b in factors]

diff = list(set(perms_in_depth) - set(products))

for _ in range(4):
    for tupol in diff[:]:
        diff.append(tup_90_rot(tupol))

xd = iterable_to_lists(diff, 0)
yd = iterable_to_lists(diff, 1)

t = np.linspace(0, 2*np.pi, 1000)
x_out = outer_radius*np.cos(t)
y_out = outer_radius*np.sin(t)
x_in = inner_radius*np.cos(t)
y_in = inner_radius*np.sin(t)

fig = go.Figure()
fig.update_layout(template='plotly_dark')
fig.add_trace(go.Scatter(x=x_out, y=y_out, mode='lines'))
fig.add_trace(go.Scatter(x=x_in, y=y_in, mode='lines'))
fig.add_trace(go.Scattergl(x=xd, y=yd, mode='markers', marker=dict(size=3)))
fig.update_yaxes(scaleanchor="x", scaleratio=1)
fig.show()
