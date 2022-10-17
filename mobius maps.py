import numpy as np
import time
import plotly.graph_objects as go
from scipy.linalg import fractional_matrix_power
from indras_functions import *
from time import perf_counter
from matplotlib import rcParams
rcParams['text.usetex'] = True
rcParams['text.latex.preamble'] = r'\usepackage{{amsmath}}'
ltx = r'$\begin{pmatrix} a & b \\ d & e \end{pmatrix}$'

t_start = perf_counter()
# Annuli
#graph_annuli()

# Make sphere
s_u, s_v, s_w = make_sphere(1, 0, 0, 0)

# Set-up orbits
mode = 'quality'  # 'quantity'
steps = 100
points = make_circle_grid(.5, 5, 5, np.pi/5, mode='circles')
a = 0
b = 1
c = 1
d = 0
a_, b_, c_, d_ = mobius_steps(a=a, b=b,
                          c=c, d=d,
                          steps=steps)

f_ps = mobius_fixed_points(a, b, c, d, 'both')
if mode == 'quantity':
    # Quantity
    orbits = make_orbits(points, lambda xy: mobius_trans(xy, a_, b_, c_, d_), steps*5, True)
elif mode == 'quality':
    # Quality
    qual_p = cx2xy(np.array([[-.1 + 1j], [.1 + 1j], [1 + .5j], [-1 + .5j]]))
    qual_p = make_circle_grid(.1, 3, 3, 2*np.pi/8, mode='circles')
    orbits = make_orbits(qual_p, lambda xy: mobius_trans(xy, a_, b_, c_, d_), steps*100, True)
else:
    raise ValueError

orbits3d = np.concatenate((orbits, np.zeros([orbits.shape[0], orbits.shape[1], 1])), axis=2)
# Project into sphere
projections = np.empty([orbits.shape[0], orbits.shape[1],  3])
for k in range(orbits.shape[0]):
    projections[k] = plane2sphere(orbits[k][:, 0], orbits[k][:, 1])

fig = go.Figure()

if mode == 'quality':
    graph_orbits(fig, orbits3d, show_end=False, orbit_color='rgba(255, 0, 0, 0.5)', orb_width=3, extra_points=f_ps, L=5)

    graph_orbits(fig, projections, show_end=False, orbit_color='rgba(0, 0, 255, 0.5)', orb_width=3, orbit_opacity=.8, L=5)

    fig.add_trace(go.Surface(x=s_u, y=s_v, z=s_w, opacity=.1,
                             colorscale=[[0, 'rgba(255, 255, 255, .1)'], [1, 'rgba(255,255,255, .1)']],
                             showscale=False))
if mode == 'quantity':
    pass

text, mtext = mobius_text(a, b, c, d, unimodular=True)
fig.update_layout(title_text=text,  font=dict(family="Century", size=15))
fig.add_annotation(text=mtext, xref="paper", yref="paper", x=0.0, y=0.0, showarrow=False, font=dict(size=30))
fig.show()

"""
# Rotate spherical projection
rot_projs = np.empty([orbits.shape[0], orbits.shape[1],  3])
for k in range(orbits.shape[0]):
    rot_proj = sphere_rotation(projections[k][:, 0], projections[k][:, 1], projections[k][:, 2],
                               axis='y', angle=np.pi/4)
    # Remove real part
    rot_projs[k] = rot_proj[:, 1:]
fig2 = go.Figure()
graph_orbits(fig2, rot_projs, show_end=True)

back2plane = np.empty([orbits.shape[0], orbits.shape[1],  3])
for k in range(orbits.shape[0]):
    back2plane[k] = sphere2plane(rot_projs[k][:, 0], rot_projs[k][:, 1], rot_projs[k][:, 2])

graph_orbits(fig2, back2plane, show_end=True)

fig2.add_trace(go.Surface(x=s_u, y=s_v, z=s_w,
                         opacity=.1,
                         colorscale=[[0, 'rgba(0, 0, 0, .5)'], [1, 'rgba(255,255,255, .5)']],
                         showscale=False))
fig2.show()
"""
t_end = perf_counter()
print(f"Total runtime: {t_end - t_start}")