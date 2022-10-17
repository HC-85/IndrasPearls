import matplotlib.pyplot as plt
import numpy as np
from numpy import pi
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.collections import PatchCollection
from matplotlib.collections import LineCollection


def rot(x, y, rads, inv=False):
    asd = [complex(a[0], a[1]) for a in zip(x, y)]
    if not inv:
        asd = [np.exp(1j * rads) * a for a in asd]
    else:
        asd = [np.exp(1j * -rads) * a for a in asd]
    x, y = zip(*[(n.real, n.imag) for n in asd])
    return list(x), list(y)


def t_tri(x, y, inv=False):
    if not inv:
        return np.array(x)+np.sqrt(3), np.array(y)
    else:
        return np.array(x)-np.sqrt(3), np.array(y)


def s_tri(x, y, inv=False):
    if not inv:
        return np.array(x)+np.sqrt(3)/2, np.array(y)+3/2
    else:
        return np.array(x)-np.sqrt(3)/2, np.array(y)-3/2


s = 6
p = [np.exp(1j*2*pi*k/s) for k in range(2)]
x, y = zip(*[(n.real, n.imag) for n in p])
x, y = list(x), list(y)
x.extend([0, x[0]])
y.extend([0, y[0]])
x = np.array(x)
y = np.array(y)
z = np.stack((x, y), axis=-1)

fig, ax = plt.subplots()
triangles1 = []
triangles2 = []
wedges = []

L_dummy = range(-6, 7, 1)
L = [(a, b) for a in L_dummy for b in L_dummy]
for rot_ang in range(30, 390, 60):
    zrot = np.stack((rot(z[:, 0], z[:, 1], rot_ang*pi/180)), axis=-1)
    for ss, tt in L:
        z_trans = zrot
        if ss > 0:
            for sss in range(ss):
                z_trans = np.stack((s_tri(z_trans[:, 0], z_trans[:, 1])), axis=-1)
        if ss < 0:
            for sss in range(ss, 0, 1):
                z_trans = np.stack((s_tri(z_trans[:, 0], z_trans[:, 1], 1)), axis=-1)

        if tt > 0:
            for ttt in range(tt):
                z_trans = np.stack((t_tri(z_trans[:, 0], z_trans[:, 1])), axis=-1)
        if tt < 0:
            for ttt in range(tt, 0, 1):
                z_trans = np.stack((t_tri(z_trans[:, 0], z_trans[:, 1], 1)), axis=-1)

        triangle1 = mpatches.Polygon(z_trans)
        triangles1.append(triangle1)

        triangle2 = mpatches.Polygon(z_trans)
        triangles2.append(triangle2)

        for i in range(3):
            wedge = mpatches.Wedge(z_trans[i, :], 1/3, (120 + 120*i + rot_ang) % 360, (180 + 120*i + rot_ang) % 360)
            wedges.append(wedge)

triangle_collection1 = PatchCollection(triangles1, zorder=1)
ax.add_collection(triangle_collection1)
triangle_collection1.set_color('green')

triangle_collection2 = PatchCollection(triangles2, zorder=2)
ax.add_collection(triangle_collection2)
triangle_collection2.set_color('blue')
triangle_collection2.set_facecolor('none')
triangle_collection2.set_linewidth(10)

wedges_collection = PatchCollection(wedges, zorder=3)
ax.add_collection(wedges_collection)
wedges_collection.set_color('red')

plt.axis('equal')
plt.xlim([-4, 4])
plt.ylim([-4, 4])
plt.axis('off')
plt.tight_layout()

plt.show()