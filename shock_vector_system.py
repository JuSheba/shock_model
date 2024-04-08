"""
-------------------------------------------------------------------------------
| A module to calculate two 3D vectors in Cartesian coordinate system z and r | 
| for every point of Cartesian orbit. The vectors are described               |
| in DOI 10.1086/177820                                                       |
| At the point (0,0,0) there is a Be star, at the orbital point there is      |
| a pulsar. The symmetry axis Z between the centres of the sources is         |
| directed from Be to the pulsar. The R axis is chosen as being under pi/2    |
| radians to the Z axis and lying in the plane of the z vector of the new     |
| system and the z vector of the Cartesian coordinate system.                 |                                           
-------------------------------------------------------------------------------
 Uses the module cartesian_orbit to get the orbit in the form (x,y,z).

 Output data in vector_z.csv, vector_r.csv :
 The vectors coordinates (x, y).

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartesian_orbit


def get_unit_vector(point_1, point_2):
    """
    Compute the unit vector pointing from point_1 to point_2.
    """
    vec = point_2 - point_1
    unit_vec = vec / np.linalg.norm(vec)
    return unit_vec


def get_normal_vector(vec_1, vec_2):
    """
    Compute the normal vector to the plane defined by vec_1 and vec_2.
    """
    normal_vec = np.cross(vec_1, vec_2)
    return normal_vec


def get_cos(unit_vec_1, unit_vec_2):
    """
    Compute the cosine of the angles between pairs of unit vectors.
    """
    return np.dot(unit_vec_1, unit_vec_2)


def create_plot(point_1, point_2, unit_vec_z, unit_vec_r):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    xx = unit_vec_z[0]
    xy = unit_vec_r[0]

    ax.quiver(*point_1, *xx, color='r', label=r'$\overline{z}$')
    ax.quiver(*point_1, *xy, color='g', label=r'$\overline{r}$')

    ax.scatter(*point_1, color='black', label='Be Star')
    ax.scatter(*point_2[0], color='blue', label='Pulsar')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.legend()
    plt.show()


def get_shock_vecs(coords, drawing):
    """
    Compute the unit vectors for all pulsar's coordinates.
    """
    result_vec_z = np.empty_like(coords)
    result_vec_r = np.empty_like(coords)
    for i in range(coords.shape[0]):
        point = coords[i]
        unit_vec_z = get_unit_vector(point, np.array([[0, 0, 0]]))
        normal_vec = get_normal_vector(unit_vec_z, np.array([[0, 0, 1]]))
        unit_vec_r = get_normal_vector(unit_vec_z, normal_vec)
        result_vec_z[i] = unit_vec_z
        result_vec_r[i] = - unit_vec_r
        cos_angle = get_cos(unit_vec_z, unit_vec_r.T)
        if cos_angle > 10e-10:
            print('Warning! The angle between unit vectors is not pi/2 rad, check your results.')
            break
        if drawing and i==1:
            create_plot(point, np.array([[0, 0, 0]]), unit_vec_z, unit_vec_r)
    return result_vec_z, result_vec_r


def run():
    try:
        df = pd.read_csv('orbit_coords.csv')
        coords_cartesian = df.values

    except FileNotFoundError:
        cartesian_orbit.run3d(False)
        df = pd.read_csv('orbit_coords.csv')
        coords_cartesian = df.values

    result_vec_z, result_vec_r = get_shock_vecs(coords_cartesian, False)

    header_z = "vec_z_x, vec_z_y, vec_z_z"
    header_r = "vec_r_x, vec_r_y, vec_r_z"

    np.savetxt('vector_z.csv', result_vec_z, delimiter=',', header=header_z, comments='')
    np.savetxt('vector_r.csv', result_vec_r, delimiter=',', header=header_r, comments='')


if __name__ == "__main__":
    run()
