"""
------------------------------------------------------------------------------
| A module to calculate  the points (x, y, z) in Cartesian coordinate system | 
| by the value of orbital elements and by the true anomaly.                  |                                             
------------------------------------------------------------------------------
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import kepler_orbit


conf={}


def get_cart_coords_from_kepler(true_anomaly, e, a, inclination, arg_periapsis, longitude):
    """
    The function calculates the points (x, y, z) in Cartesian coordinate system by the value 
    of orbital elements and by the true anomaly. These formulas are essentially multiplication 
    of 3 rotation matrices, the original z coordinate was zero.

    Parameters
    ----------
    true_anomaly : 1D numpy.ndarray
        the true anomaly calculated using the Kepler equation, nu
    e : float
        the eccentricity of the orbit (0,1)
    a : float
        semi-major axis of the orbit (AU)
    inclination : float
        inclination of the orbit, i (degrees)
    longitude: float
        longitude of the ascending node, Omega (degrees)
    arg_periapsis: float
        Argument of periapsis, omega (degrees)

    Returns
    -------
    coords_cartesian :  2D numpy.ndarray
    """
    radius = a * (1.0 - e**2) / (1.0 + e * np.cos(true_anomaly))
    alpha = arg_periapsis + true_anomaly

    x_cartesian = radius * (np.cos(longitude)*np.cos(alpha) -
                            np.sin(longitude) * np.sin(alpha) * np.cos(inclination))
    y_cartesian = radius * (np.sin(longitude)*np.cos(alpha) +
                            np.cos(longitude) * np.sin(alpha) * np.cos(inclination))
    z_cartesian = radius * (np.sin(inclination) * np.sin(alpha))
    coords_cartesian = np.vstack([x_cartesian, y_cartesian, z_cartesian])
    return coords_cartesian


def create_animation(coords, n_points):
    """
    Draws an animation of the pulsar's orbital motion
    """
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    x_min, x_max = np.min(coords[0, :]), np.max(coords[0, :])
    y_min, y_max = np.min(coords[1, :]), np.max(coords[1, :])
    z_min, z_max = np.min(coords[2, :]), np.max(coords[2, :])
    ax.set(xlim3d=(x_min-1, x_max+1), xlabel='X, AU')
    ax.set(ylim3d=(y_min-1, y_max+1), ylabel='Y, AU')
    ax.set(zlim3d=(z_min-1, z_max+1), zlabel='Z, AU')

    ax.plot(coords[0, :], coords[1, :], coords[2, :])
    ax.plot(0, 0, 0, 'o', color='gold', markersize=4)

    point = ax.plot([], [], 'o', color='r', markersize=4)[0]

    def update_point(num, x, y, z, point):
        point.set_data([x[num - 1]], [y[num - 1]])
        point.set_3d_properties(z[num - 1])
        return point

    ani = FuncAnimation(fig, update_point, n_points, 
                        fargs=(coords[0, :], coords[1, :], coords[2, :], point), interval=1)
    plt.show()


def init_conf():
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', default='param_conf.json', help='Configuration path')

    args = parser.parse_args()
    global conf
    if args.conf:
        with open(args.conf, 'r', encoding="utf-8") as f:
            conf = json.load(f)


def run3d(animate):
    """
    The function calculates the (x, y, z) coordinates for the orbit. 
    Draws an animation of the orbit motion if animate = True.
    """
    init_conf()
    e = conf['orbit']['e']
    a = conf['orbit']['a']

    inclination = np.radians(conf['orbit']['inclination'])
    longitude = np.radians(conf['orbit']['longitude'])
    arg_periapsis = np.radians(conf['orbit']['arg_periapsis'])

    n_points, true_anomaly, *_ = kepler_orbit.run2d(False)
    coords_cartesian = get_cart_coords_from_kepler(true_anomaly, e, a, inclination,
                                                   arg_periapsis, longitude)
    header = "x_orbit, y_orbit, z_orbit"
    np.savetxt('orbit_coords.csv', coords_cartesian.T, delimiter=',', header=header, comments='')
    
    if animate:
        create_animation(coords_cartesian, n_points)


if __name__ == "__main__":
    run3d(True)
