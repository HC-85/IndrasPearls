import plotly.graph_objects as go
import time
import pandas as pd
import numpy as np
from numpy import pi


def t_hex(x, y, inv=False):
    if not inv:
        return np.array(x)+np.sqrt(3), np.array(y)
    else:
        return np.array(x)-np.sqrt(3), np.array(y)


def s_hex(x, y, inv=False):
    if not inv:
        return np.array(x)+np.sqrt(3)/2, np.array(y)+3/2
    else:
        return np.array(x)-np.sqrt(3)/2, np.array(y)-3/2


def rot(x, y, rads, inv=False):
    asd = [complex(a[0], a[1]) for a in zip(x, y)]
    if not inv:
        asd = [np.exp(1j * rads) * a for a in asd]
    else:
        asd = [np.exp(1j * -rads) * a for a in asd]
    x, y = zip(*[(n.real, n.imag) for n in asd])
    return list(x), list(y)

s = 6
p = [np.exp(1j*2*pi*k/s) for k in range(s)]
x, y = zip(*[(n.real, n.imag) for n in p])
x, y = list(x), list(y)
x.append(x[0])
y.append(y[0])

x, y = rot(x, y, pi/6)

fig = go.FigureWidget()
fig.update_layout(template='plotly_dark')
fig.update_traces(line_width=1)
fig.update_yaxes(scaleanchor="x", scaleratio=1)
L = 1
for t in range(-L, L):
    for s in range(-L, L):
        X, Y = x, y
        if t > 0:
            for tt in range(t):
                X, Y = t_hex(X, Y)
        elif t < 0:
            for tt in range(t, 0):
                X, Y = t_hex(X, Y, True)
        if s > 0:
            for ss in range(s):
                X, Y = s_hex(X, Y)
        elif s < 0:
            for ss in range(s, 0):
                X, Y = s_hex(X, Y, True)
        time.sleep(.5)
        fig.add_trace(go.Scatter(x=X, y=Y, mode='lines', fill='tonexty', line_color='lightseagreen'))
fig.show()