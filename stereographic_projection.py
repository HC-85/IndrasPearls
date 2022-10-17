import matplotlib.image as img
from time import perf_counter
from indras_functions import *

t_start = perf_counter()
x_min = 1.5
y_min = -.75
y_max = .75

pix = img.imread(r'D:\PyCharm\Indras_Pearls\fox.png')
height, width, _ = pix.shape
#pix = np.swapaxes(pix, 0, 1)
pix = pix.reshape(-1, pix.shape[-1])

plane_xy = make_square_grid(x_min, x_min + (y_max-y_min)*width/height, width, y_max, y_min, height)
x, y = [*plane_xy.T]

projection = plane2sphere(x, y)

s_u, s_v, s_w = make_sphere(1, 0, 0, 0)

colors = []
for pixel in pix:
    colors.append('rgba({}, {}, {}, 1)'.format(*pixel))

rot_proj = sphere_rotation(projection[:, 0], projection[:, 1], projection[:, 2], 'x', 0)

fig = go.Figure()
fig.add_trace(go.Scatter3d(x=x,
                           y=y,
                           z=np.zeros_like(x),
                           mode='markers', marker={'size': 1, 'color': colors}, marker_symbol='square'))
fig.add_trace(go.Scatter3d(x=rot_proj[:, 1],
                           y=rot_proj[:, 2],
                           z=rot_proj[:, 3],
                           mode='markers', marker={'size': 1, 'color': colors}, marker_symbol='square'))

fig.add_trace(go.Surface(x=s_u, y=s_v, z=s_w,
                         opacity=.2,
                         colorscale=[[0, 'rgba(0, 0, 0, .5)'], [1, 'rgba(255,255,255, .5)']],
                         showscale=False))
# Corner rays
for k in [0, width-1, -width+1, -1]:
    fig.add_trace(go.Scatter3d(x=[0, x[k]],
                               y=[0, y[k]],
                               z=[1, 0],
                               mode='lines', line={'width': 2, 'color': 'rgba(255, 0, 0, 1)', 'dash': 'dash'}))

fig.update_layout(scene={'aspectmode': 'data', 'aspectratio': dict(x=1, y=1, z=1)})
fig.update_layout(template='plotly_dark', showlegend=False)
fig.show()
t_end = perf_counter()
print(f"Total runtime: {t_end - t_start}")
