"""
---------------------------------------------------------------
| From the counted shocks and vectors for the shock, computes |
| the coordinates of the shock in the 3d Cartesian system.    |
---------------------------------------------------------------
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shock_vector_system
import plane_shock


def read_data():
    df_r = pd.read_csv('vector_r.csv')
    vector_r_array = df_r.values

    df_z = pd.read_csv('vector_z.csv')
    vector_z_array = df_z.values

    df_orbit = pd.read_csv('orbit_coords.csv')
    orbit_coords_array = df_orbit.values

    df_shock = pd.read_csv('shock.csv')
    shock_array = df_shock.values

    return vector_r_array, vector_z_array, orbit_coords_array, shock_array



def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / np.sqrt(np.dot(axis, axis))
    a = np.cos(theta / 2.0)
    b, c, d = - axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])



def transform_coordinates(v1, v2, v3, origin, shock_array):
    """
    Replacing the standard Cartesian system vectors with the given new ones. 
    The shock given in the new coordinate system is shifted to the point of the pulsar's orbit.
    Parameters
    ----------
    v1 : numpy.ndarray
        First vector of the new basis.
    v2 : numpy.ndarray
        Second vector of the new basis.
    v3 : numpy.ndarray
        Third vector of the new basis.
    origin : numpy.ndarray
        Point of the pulsar's orbit to which the shock is shifted.
    shock_array : numpy.ndarray
        Shock given in the new coordinate system.

    Returns
    -------
    numpy.ndarray
        Transformed shock array shifted to the point of the pulsar's orbit.
    """
    old_basis = np.array([[1, 0, 0],
                          [0, 1, 0],
                          [0, 0, 1]])

    old_basis = np.array(old_basis)
    new_basis = np.column_stack((v1, v2, v3))

    transition_matrix = np.linalg.inv(old_basis).dot(new_basis)
    res = np.dot(transition_matrix, shock_array) + origin
    return res


def get_unit_vec(vec_1, vec_2):
    """
    Compute the unit vector pointing from point_1 to point_2.
    """
    normal_vec = np.cross(vec_2, vec_1)
    normal_vec = normal_vec / np.linalg.norm(normal_vec)
    return normal_vec



def distance_to_line_2d(x, y, line_point1, line_point2):
    """
    Calculating the distance from points (x, y) to a line 
    given by two points line_point1 and line_point2.
    """
    distance = np.abs((line_point2[0] - line_point1[0]) * (line_point1[1] - y)
                    - (line_point1[0] - x) * (line_point2[1] - line_point1[1])) / \
                       np.sqrt((line_point2[0] - line_point1[0])**2 +
                               (line_point2[1] - line_point1[1])**2)
    return distance

def run():
    try:
        vector_r_array, vector_z_array, orbit_coords_array, shock_array = read_data()

    except FileNotFoundError:
        shock_vector_system.run()
        plane_shock.run(False)
        vector_r_array, vector_z_array, orbit_coords_array, shock_array = read_data()


    orbit_index = 300

    orbit_point = orbit_coords_array[orbit_index]

    vector_z = vector_z_array[orbit_index]
    vector_r = vector_r_array[orbit_index]
    vector_n = - get_unit_vec(vector_r, vector_z)


    shock_array_with_z = np.column_stack((shock_array, np.zeros(len(shock_array))))
    transformed_shock_array = []
    for i in range(shock_array_with_z[:,0].size):
        transformed_shock_array1 = transform_coordinates(vector_r,
                                                    vector_z,
                                                    vector_n,
                                                    orbit_point, shock_array_with_z[i,:])
        transformed_shock_array.append(transformed_shock_array1)

    transformed_shock_array = np.array(transformed_shock_array)

    angles = np.linspace(0, 2*np.pi, 50)
    mtrx = rotation_matrix(vector_z, np.pi/2)

    final_shock = np.dot(transformed_shock_array, mtrx)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    combined_array = np.empty((0, 3))
    for angle in angles:
        mtrx = rotation_matrix(vector_z, angle)
        final_shock = np.dot(transformed_shock_array, mtrx)
        combined_array = np.vstack((combined_array, final_shock))
        ax.plot(final_shock[:, 0], final_shock[:, 1], final_shock[:, 2],
                label='Shock2', color='b', alpha=0.2)

    ax.quiver(orbit_point[0], orbit_point[1], orbit_point[2],
              vector_r[0], vector_r[1], vector_r[2], color='r', label=r'$\overline{r}$')
    ax.quiver(orbit_point[0], orbit_point[1], orbit_point[2],
              vector_z[0], vector_z[1], vector_z[2], color='g', label=r'$\overline{z}$')
    ax.quiver(orbit_point[0], orbit_point[1], orbit_point[2],
              vector_n[0], vector_n[1], vector_n[2], color='b', label=r'$\overline{n}$')


    ax.scatter(orbit_point[0], orbit_point[1], orbit_point[2], color='b', label='Pulsar')
    ax.scatter(0, 0, 0, color='b', label='Be star')


    ax.plot(orbit_coords_array[:, 0], orbit_coords_array[:, 1], orbit_coords_array[:, 2],
            color='orange', label='orbit_array')


    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim([-7, 7])
    ax.set_ylim(-13,1)
    ax.set_zlim(-8, 5)
    plt.show()


run()
