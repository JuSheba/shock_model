# Shock Model

- kepler_orbit: A module to calculate the true anomaly for different values of time.
- cartesian_orbit: A module to calculate  the points (x, y, z) in Cartesian coordinate system by the value of orbital elements and by the true anomaly. Uses kepler_orbit. 
- shock_vector_system: A module to calculate two 3D vectors in Cartesian coordinate system z and r for every point of Cartesian orbit. The vectors are described in DOI 10.1086/177820. Uses cartesian_orbit.
- plane_shock: A module for computing the shape of the shock in the plane.
- main: From the counted shocks and vectors for the shock, computes the coordinates of the shock in the 3d Cartesian system. Uses shock_vector_system and plane_shock.
- param_conf: A file with all orbit and shock parameters.

### To use this module please run the following commands:

`pip install --upgrade pip`

`pip install numpy`

`pip install pandas`

`pip install matplotlib`

`pip install pandas`

`pip install argparse`
