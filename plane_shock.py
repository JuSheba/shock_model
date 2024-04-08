"""
-------------------------------------------------------------
| Module for computing the shape of the shock in the plane. |
| See DOI 10.1086/177820                                    |
-------------------------------------------------------------

 Input data in param_conf.json :
 init_mass_rate_first   =   The rate of mass loss of the Be star,
 init_mass_rate_second  =   The rate of mass loss of the pulsar,
 velocity_first         =   The velocity of the plane wind (Be star),
 velocity_second        =   The velocity of the plane wind (pulsar),
 theta_start            =   Init angle for computing.
 theta_end              =   Final angle for computing.

 Output data in shock.csv :
 The shock coordinates (x, y).
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt


conf={}


def calculate_beta(init_mass_rate_first, init_mass_rate_second,
                   velocity_first, velocity_second):
    """
    Look at equation 24 in the aforementioned article.
    """
    beta = init_mass_rate_first * velocity_first / \
           (init_mass_rate_second * velocity_second)
    return beta


def calculate_angle_from_theta(theta, beta):
    """
    Look at equation 26 in the aforementioned article.
    """
    under_sqrt = 1.0 + 0.8 * beta * (1.0 - theta / np.tan(theta))
    theta_1 = np.sqrt(7.5 * ( -1.0 + np.sqrt(under_sqrt)))
    return theta_1


def calculate_locus_from_angles(distance, theta, theta_1):
    """
    Look at equation 23 in the aforementioned article.
    """
    locus = distance * np.sin(theta_1) / np.sin(theta + theta_1)
    return locus


def get_cart_coords_from_polar(angle, radius):
    x_coord = radius * np.cos(angle)
    y_coord = radius * np.sin(angle)
    return x_coord, y_coord


def workaround(x_coord, y_coord):
    x_coord = x_coord[:-round(x_coord.size/10)]
    y_coord = y_coord[:-round(y_coord.size/10)]
    return x_coord, y_coord


def create_animation(x_array, y_array, y_interpolated, beta):
    plt.scatter(x_array, y_array, label=f'Data for beta = {beta}', s=1)
    plt.plot(x_array, y_interpolated, label='Fitted Curve', color='red')
    plt.xlabel('R')
    plt.ylabel('Z')
    plt.legend()
    plt.grid(True)
    plt.show()


def init_conf():
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', default='param_conf.json', help='Configuration path')

    args = parser.parse_args()
    global conf
    if args.conf:
        with open(args.conf, 'r', encoding="utf-8") as f:
            conf = json.load(f)


def run(draw):
    init_conf()
    init_mass_rate_first = conf['shock']['init_mass_rate_first']
    init_mass_rate_second = conf['shock']['init_mass_rate_second']
    velocity_first = conf['shock']['velocity_first']
    velocity_second = conf['shock']['velocity_second']
    theta_start = conf['shock']['theta_start']
    theta_end = conf['shock']['theta_end']

    theta = np.linspace(theta_start, theta_end * np.pi, pow(10, 5))

    beta = calculate_beta(init_mass_rate_first, init_mass_rate_second,
                   velocity_first, velocity_second)
    beta = 0.03125   # only for example
    theta_1 = calculate_angle_from_theta(theta, beta)
    locus = calculate_locus_from_angles(1, theta, theta_1)

    x_coord, y_coord = get_cart_coords_from_polar(theta, locus)
    x_coord, y_coord = workaround(x_coord, y_coord)
    all_coords = np.vstack([y_coord, x_coord])

    header_coords = "x_coord, y_coord"
    np.savetxt('shock.csv', all_coords.T, delimiter=',', header=header_coords, comments='')

    if draw:
        betas = [0.5, 0.25, 0.125, 0.0625, 0.03125]
        fig, ax = plt.subplots()
        for beta in betas:
            theta_1 = calculate_angle_from_theta(theta, beta)
            locus = calculate_locus_from_angles(1, theta, theta_1)
            x_coord = locus * np.cos(theta)
            y_coord = locus * np.sin(theta)
            ax.scatter(x_coord, y_coord, label=f'Beta = {beta}', s=0.5)
            ax.scatter(0.0, 0.0, s=50, color='black')
        ax.legend()

        ax.set_xlabel('R')
        ax.set_ylabel('Z')
        ax.set_xlim((-1.1, 0.6))
        ax.set_ylim((-0.125, 1.6))
        plt.show()


if __name__ == "__main__":
    run(True)
