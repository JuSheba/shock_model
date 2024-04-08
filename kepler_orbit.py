"""
-----------------------------------------------------------------------------------
| A module to calculate the true anomaly for different values of time.            |
| It takes the Kepler equation for the case of an elliptical orbit, solves        |
| it using the Newton-Raphson method and finds the true anomaly from the E values |                                              
-----------------------------------------------------------------------------------
Input data in param_conf.json :
eps      =   Angle calculation accuracy * 
period   =   Orbital period of the pulsar (in days)
e        =   Eccentricity (0,1)
a        =   Major semi-axis (AU)

animate: True or False for animation

Used in the calculation process:
ecc_anomaly   =   Eccentric anomaly (E)
init_ecc      =   Initial approximation for E
mean_anomaly  =   Mean anomaly (M)
true_anomaly  =   True anomaly [0, 2pi)
t             =   Number of time array steps

Output data (len = n_points):
true_anomaly_array
time_array = Time from 0 to T
x_array    = X coord (AU)
y_array    = Y coord (AU)


* Since the elliptical velocity varies greatly depending on the point in the orbit,
 not every time array step is suitable. By choosing eps, you set the maximum difference 
 in radians between 2 points on the ellipse (def get_perfect_time).
"""

import json
import argparse
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


conf={}


def kepler_equation(ecc_anomaly, mean_anomaly, e):
    """
    f(E) = E - e * sinE - M = 0
    """
    return ecc_anomaly - mean_anomaly - e * np.sin(ecc_anomaly)

def d_kepler(ecc_anomaly, mean_anomaly, e):
    """
    f'(E), first derivative of Kepler's eq.
    """
    return 1 - e * np.cos(ecc_anomaly)

def calculate_mean_anomaly(period, t):
    """
    M = 2pi * (t - t0) / T
    t0 - time at which the body is at the pericenter
    """
    time_array = np.linspace(0, period, t)
    mean_anomaly = (2 * np.pi * time_array) / period
    return mean_anomaly

def calculate_true_anomaly(ecc_anomaly, e):
    """
    tg(v / 2)  = sin(v) / (1 + cos(v)) = tg(E/2) * sqrt((1 + e) / (1 - e))

    is recalculated for values from 0 to 2pi radians, 
    because the arctan/arctan2 function gives other values
    """
    true_anomaly = 2 * np.arctan2(np.sqrt((1 + e) / (1 - e)) * np.tan(ecc_anomaly / 2), 1)
    true_anomaly = (true_anomaly + 2 * np.pi) % (2 * np.pi)
    return true_anomaly

def solve_kepler_equation(mean_anomaly, e):
    """
    Calculated initial approximation for E and solves kepler's eq. using the Newton-Raphson method
    """
    init_ecc = mean_anomaly + e * np.sin(mean_anomaly)
    ecc_anomaly = optimize.newton(kepler_equation, init_ecc,
                                  fprime=d_kepler, args=(mean_anomaly, e))
    return ecc_anomaly

def get_perfect_time(period, eps, e):
    """
    Calculates a step for the time array such that the maximum difference 
    in radians between neighboring orbit points is not more than eps
    """
    t = 100
    mean_anomaly_array = calculate_mean_anomaly(period, t)
    true_anomaly_array = get_true_from_mean(mean_anomaly_array, e)
    while true_anomaly_array[1] - true_anomaly_array[0] > eps:
        t = t + 100
        mean_anomaly_array = calculate_mean_anomaly(period, t)
        true_anomaly_array = get_true_from_mean(mean_anomaly_array, e)
    return t, true_anomaly_array

def get_true_from_mean(mean_anomaly_array, e):
    """
    Сalculates the true anomaly array from the mean anomaly array
    """
    true_anomaly_array = []
    for mean_anomaly in mean_anomaly_array:
        ecc_anomaly = solve_kepler_equation(mean_anomaly, e)
        true_anomaly = calculate_true_anomaly(ecc_anomaly, e)
        true_anomaly_array.append(true_anomaly)
    return true_anomaly_array


def get_coords_for_true(true_anomaly_array, e, a, b):
    """
    Сalculates the x_coord and y_coord arrays from the true anomaly array
    """
    x_array = a * (np.cos(true_anomaly_array) - e)
    y_array = b * np.sin(true_anomaly_array)
    return x_array, y_array


class AnimatedPulsar:
    def __init__(self, ax, true_anomaly_array, a, b, e, interval):
        self.ax = ax
        self.true_anomaly_array = true_anomaly_array
        self.a = a
        self.b = b
        self.e = e
        self.interval = interval

        self.vector, = self.ax.plot([], [], color='slategray', linestyle='-', label="Position vector")
        self.point, = self.ax.plot([], [], 'o', color='r', label="Pulsar")
        self.text = self.ax.text(0.05, 0.90, '', transform=self.ax.transAxes, fontsize=12, color='black')

        self.ani = FuncAnimation(self.ax.figure, self.update, frames=len(self.true_anomaly_array),
                                 init_func=self.init, blit=True, interval=self.interval)

    def init(self):
        self.vector.set_data([], [])
        self.point.set_data([], [])
        self.text.set_text('')
        return self.vector, self.point, self.text

    def update(self, frame):
        x = self.a * (np.cos(self.true_anomaly_array[frame]) - self.e)
        y = self.b * np.sin(self.true_anomaly_array[frame])

        self.vector.set_data([0, x], [0, y])
        self.point.set_data([x], [y])
        self.text.set_text(f"v: {np.degrees(self.true_anomaly_array[frame]):.1f}°\
                           \nTime:{frame + 1}/{len(self.true_anomaly_array)}")
        return self.vector, self.point, self.text


def create_animation(true_anomaly_array, a, b, e):
    """
    Draws an animation of the pulsar's orbital motion
    """
    fig, ax = plt.subplots()
    ax.set_xlim(-1.5 * a, 1.5 * a)
    ax.set_ylim(-1.5 * b, 1.5 * b)
    pulsar_animation = AnimatedPulsar(ax, true_anomaly_array, a, b, e, 1)
    x, y = get_coords_for_true(true_anomaly_array, e, a, b)

    plt.grid(color='silver', alpha=0.5)
    plt.plot(x,y)
    plt.plot([0], [0], 'o', color='gold')
    plt.xlabel("x, AU")
    plt.ylabel("y, AU")
    plt.legend()
    plt.axis('equal')
    plt.show()


def init_conf():
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', default='param_conf.json', help='Configuration path')

    args = parser.parse_args()
    global conf
    if args.conf:
        with open(args.conf, 'r', encoding="utf-8") as f:
            conf = json.load(f)


def run2d(animate):
    """
    The function calculates the true anomaly, the optimal number of points
    and the (x, y) coordinates for the orbit. Draws an animation of the orbit 
    motion if animate = True.
    """
    init_conf()
    period = conf['orbit']['period']
    e = conf['orbit']['e']
    a = conf['orbit']['a']
    b = conf['orbit']['b']
    eps =  conf['orbit']['eps']

    n_points, true_anomaly_array = get_perfect_time(period, eps, e)
    time_array = np.linspace(0, period, n_points)
    x_array, y_array = get_coords_for_true(true_anomaly_array, e, a, b)
    if animate:
        create_animation(true_anomaly_array, a, b, e)
    return n_points, true_anomaly_array, time_array, x_array, y_array


if __name__ == "__main__":
    run2d(True)
