import math
import numpy as np
from plotly import graph_objs as go
import quaternion
from scipy.linalg import fractional_matrix_power


def make_square_grid(x_min, x_max, x_n, y_min, y_max, y_n):
    x, y, = np.meshgrid(np.linspace(x_min, x_max, x_n), np.linspace(y_min, y_max, y_n))
    cart_prod = np.dstack((x.ravel(), y.ravel())).reshape(-1, 2)
    return cart_prod


def make_circle_grid(inner_rad, outer_rad, r_n, theta_grid, mode='rays', x=0, y=0):
    if mode == 'rays':
        rays = cx2xy(np.linspace(inner_rad, outer_rad, r_n))
        for k in np.linspace(2 * np.pi / theta_grid, 2 * np.pi, theta_grid):
            new_ray = np.exp(1j * k) * np.linspace(inner_rad, outer_rad, r_n)
            rays = np.concatenate((rays, cx2xy(new_ray)), axis=0)
        x = x*np.ones(len(rays))
        y = y*np.ones(len(rays))
        rays = np.dstack((rays[:, 0]+x, rays[:, 1]+y)).reshape(-1, 2)
        return rays
    if mode == 'circles':
        for k in np.linspace(inner_rad, outer_rad, r_n):
            if k == inner_rad:
                circles = cx2xy(k * np.exp(1j * np.arange(0, 2 * np.pi, theta_grid / k)))
                continue
            new_circle = k * np.exp(1j * np.arange(0, 2*np.pi, theta_grid/k))
            circles = np.concatenate((circles, cx2xy(new_circle)), axis=0)
        x = x * np.ones(len(circles))
        y = y * np.ones(len(circles))
        circles = np.dstack((circles[:, 0] + x, circles[:, 1] + y)).reshape(-1, 2)
        return circles


def make_ring(radius, n_theta=100, x=0, y=0):
    theta = np.linspace(0, 2*np.pi, n_theta)
    ring = radius*np.exp(1j*theta) + complex(x, y)
    return cx2xy(ring)


def make_sphere(radius, x=0, y=0, z=0, n_azimutal=100, n_polar=100):
    azimutal = np.linspace(0, -2 * np.pi, n_azimutal)
    polar = np.linspace(0, np.pi, n_polar)
    u = radius * np.outer(np.cos(polar), np.sin(azimutal)) + x
    v = radius * np.outer(np.sin(polar), np.sin(azimutal)) + y
    w = np.outer(radius * np.ones_like(polar), np.cos(azimutal)) + z
    return u, v, w


def xy2cx(xy):
    z = xy[:, 0] + 1j*xy[:, 1]
    return z


def cx2xy(z):
    x = z.real
    y = z.imag
    return np.dstack((x, y)).reshape(-1, 2)


def sphere2plane(u, v, w):
    x = u / (1 - w)
    y = v / (1 - w)
    z = np.zeros_like(x)
    return np.stack((x, y, z)).T


def plane2sphere(x, y):
    k = x**2 + y**2
    u = 2*x / (k + 1)
    v = 2*y / (k + 1)
    w = (k - 1) / (k + 1)
    return np.stack((u, v, w)).T


def mobius_trans(z, a, b, c, d, inverse=False):
    if any(z == np.inf):
        ind = np.where(z == np.inf)
        if c != 0:
            z[ind] = a/c
        else:
            z[ind] = np.inf
    if a*d - b*c == 0:
        raise ValueError('Determinant cannot be zero')
    if not inverse:
        w = (a*z + b)/(c*z + d)
    else:
        w = (d*z - b)/(-c*z + a)
    return w


def make_orbits(xy, trans, n_steps, swapaxes=False):
    stack = xy
    for i in range(n_steps):
        xy_trans = cx2xy(trans(xy2cx(xy)))
        stack = np.dstack((stack, xy_trans))
        xy = xy_trans
    if swapaxes:
        stack = stack.swapaxes(1, 2)
    return stack


def mobius_steps(a, b, c, d, steps):
    matrix = np.array([[a, b], [c, d]])
    matrix_step = fractional_matrix_power(matrix, 1 / steps)
    a, b, c, d = matrix_step.ravel()
    return a, b, c, d

# X AND Y AXIS WRONG
def sphere_rotation(x, y, z, axis, angle):
    p = quaternion.as_quat_array(np.stack((np.ones_like(x), x, y, z), axis=1))
    if axis == 'y':
        q = np.quaternion(np.cos(angle), np.sin(angle), 0, 0)
        q_inv = np.quaternion(np.cos(-angle), np.sin(-angle), 0, 0)
    elif axis == 'x':
        q = np.quaternion(np.cos(angle), 0, np.sin(angle), 0)
        q_inv = np.quaternion(np.cos(-angle), 0, np.sin(-angle), 0)
    elif axis == 'z':
        q = np.quaternion(np.cos(angle), 0, 0, np.sin(angle))
        q_inv = np.quaternion(np.cos(-angle), 0, 0, np.sin(-angle))
    else:
        raise ValueError
    return quaternion.as_float_array(q * p * q_inv)


class Annulus:
    def __init__(self, inner_rad, outer_rad, x=0, y=0):
        self.inner_rad = inner_rad
        self.outer_rad = outer_rad
        self.x = x
        self.y = y
        self.inner_xy = self.get_inner_xy()
        self.outer_xy = self.get_outer_xy()

    def draw_go(self, fig, linecolor='rgb(255, 255, 255)', fillcolor='rgba(0, 0, 255, .5)'):
        fig.add_trace(go.Scatter(x=self.inner_xy[:, 0], y=self.inner_xy[:, 1], mode='lines',
                                 line={"color": linecolor}))
        fig.add_trace(go.Scatter(x=self.outer_xy[:, 0], y=self.outer_xy[:, 1], mode='lines',
                                 line={"color": linecolor},
                                 fillcolor=fillcolor,
                                 fill='tonexty'))

    def get_inner_xy(self, theta_n=500):
        return make_ring(self.inner_rad, theta_n, x=self.x, y=self.y)

    def get_outer_xy(self, theta_n=500):
        return make_ring(self.outer_rad, theta_n, x=self.x, y=self.y)


def graph_annuli(fig, func, steps, radii=None, colors=None):
    if colors is None:
        colors = ['rgba(255, 0, 0, .5)', 'rgba(0, 255, 0, .5)', 'rgba(0, 0, 255, .5)', 'rgba(255, 255, 0, .5)']
    if radii is None:
        radii = [*np.linspace(.4, .7, len(colors) + 1)]
    for n in range(len(radii)-1):
        an = Annulus(radii[n], radii[n+1], 1, 1)
        an.draw_go(fig, linecolor='rgb(255, 255, 255)', fillcolor=colors[n])
        for _ in range(steps):
            an.inner_xy = cx2xy(func(an.inner_xy))
            an.outer_xy = cx2xy(func(an.outer_xy))
            an.draw_go(fig, linecolor='rgb(255, 255, 255)', fillcolor=colors[n])
    fig.update_layout(template='plotly_dark', showlegend=False)
    fig.update_yaxes(scaleanchor='x', scaleratio=1)
    fig.show()


def graph_orbits(fig, orbits, show_start=False, show_end=False, mode='3D', orbit_color='rgba(255, 255, 255, 0.1)',
                 orbit_opacity=.3, orb_width=1, extra_points=None, L=5):
    if mode == '2D':
        for orbit in orbits:
            fig.add_trace(go.Scattergl(x=orbit[0, :], y=orbit[1, :], mode="lines",
                                    marker={"size": 1, "line": {"color": 'rgba(255, 255, 255, 0.1)'}}))
            if show_start:
                fig.add_trace(go.Scattergl(x=[orbit[0, 0]], y=[orbit[1, 0]], mode="markers",
                                          marker={"color": 'rgba(0, 0, 255, 0.8)', "size": 2}))

            if show_end:
                fig.add_trace(go.Scattergl(x=[orbit[0, -1]], y=[orbit[1, -1]], mode="markers",
                                          marker={"color": 'rgba(255, 0, 0, .7)', "size": 1}))
        fig.update_xaxes(range=[-10, 10])
        fig.update_yaxes(range=[-10, 10])
    elif mode == '3D':
        for orbit in orbits:
            if np.allclose(orbit[:, 1], 0) & np.allclose(orbit[:, 2], 0):
                continue
            fig.add_trace(go.Scatter3d(x=orbit[:, 0], y=orbit[:, 1], z=orbit[:, 2], mode="lines",
                                        marker={"size": 1},
                                        line={"color": orbit_color, "width": orb_width},
                                        opacity=orbit_opacity))
            if show_start:
                fig.add_trace(go.Scatter3d(x=[orbit[0, 0]], y=[orbit[0, 1]], z=[orbit[0, 2]], mode="markers",
                                          marker={"color": 'rgba(0, 0, 255, 0.8)', "size": 1.5},
                                          opacity=.5))

            if show_end:
                fig.add_trace(go.Scatter3d(x=[orbit[-1, 0]], y=[orbit[-1, 1]], z=[orbit[-1, 2]], mode="markers",
                                          marker={"color": 'rgba(255, 0, 0, 1)', "size": 1.5},
                                          opacity=1))
        if extra_points is not None:
            if len(extra_points) == 1:
                fig.add_trace(go.Scatter3d(x=[extra_points[0]], y=[extra_points[1]], z=np.zeros(len(extra_points)), mode="markers",
                                           marker={"color": 'rgba(0, 255, 0, 1)', "size": 3},
                                           opacity=1))
            else:
                fig.add_trace(go.Scatter3d(x=extra_points[0, :], y=extra_points[1, :], z=np.zeros(len(extra_points)),
                                           mode="markers",
                                           marker={"color": 'rgba(0, 255, 0, 1)', "size": 3},
                                           opacity=1))
        fig.update_layout(
            scene=dict(
                                    xaxis=dict(range=[-L, L]),
                                    yaxis=dict(range=[-L, L]),
                                    zaxis=dict(range=[-L, L]))
        )
    fig.update_layout(template='plotly_dark', showlegend=False)


def mobius_fixed_points(a, b, c, d, which='plus'):
    if c == 0:
        return [(b/((1-a)*d)).real, (b/((1-a)*d)).imag]
    else:
        if which == 'plus':
            fp = (a - d + np.sqrt((d - a) ** 2 + 4*b*c)) / (2 * c)
        elif which == 'minus':
            fp = (a - d - np.sqrt((d - a) ** 2 + 4*b*c)) / (2 * c)
        elif which == 'both':
            fp = np.array([(a - d + np.sqrt((d - a) ** 2 + 4*b*c)) / (2 * c),
                           (a - d - np.sqrt((d - a) ** 2 + 4*b*c)) / (2 * c)])
        else:
            raise ValueError
    return np.array([fp.real, fp.imag])


def mobius_text(a, b, c, d, unimodular=False):
    u_p, v_p = mobius_fixed_points(a, b, c, d, 'plus')
    u_m, v_m = mobius_fixed_points(a, b, c, d, 'minus')
    if (u_p == u_m) & (v_p == v_m):
        if c == 0:
            text = "Affine map<br>Fixed point: {}".format(u_p + 1j * v_p)
        else:
            text = "--- map<br>Fixed point: {}".format(u_p + 1j * v_p)
    else:
        if c == 0:
            text = "Affine map<br>Fixed points: {} and {}".format(u_p + 1j * v_p, u_m + 1j*v_m)
        else:
            text = "--- map<br>Fixed points: {} and {}".format(u_p + 1j * v_p, u_m + 1j*v_m)
    if unimodular:
        a, b, c, d = np.array([a, b, c, d])/np.sqrt(math.copysign(a*d-b*c, 1))
        mtext = r'<br>$\begin{pmatrix}' + r'{} & {} \\ {} & {}'.format(a, b, c, d) + r'\end{pmatrix}$'
        return text, mtext
    return text


def circle_inversion(z, a, r):
    """z is the point to invert
       a is the centre of the circle
       r is the radius of the circle
       rz + a moves the unit circle
       1/z* inverts it"""
    return a + r**2/(z-a)


def make_polystar(sides=5, skips=2, radius=1, x=0, y=0, rot=np.pi/5):
    v = radius*np.exp(1j*2*np.pi*np.linspace(1, sides, sides)/sides)*np.exp(1j*rot)
    skipsmult = np.arange(0, skips*sides, skips)
    if all(np.array([*map(lambda mult: math.gcd(mult, sides), skipsmult[1:])]) == 1):
        skipsmult = np.append(skipsmult, skipsmult[0])
        v = v[skipsmult % sides]
        x = x * np.ones(len(v))
        y = y * np.ones(len(v))
        return v.real + x, v.imag + y
    else:
        raise ValueError('Try other skips')
