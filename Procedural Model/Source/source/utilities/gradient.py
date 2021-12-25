from scipy.interpolate import RectBivariateSpline
import numpy as np

def minimal_path(travel_time, starting_point, dx, boundary, steps, N=100):
    """
    Find the optimal path from starting_point to the zero contour
    of travel_time.

    Solve the equation x_t = - grad t / | grad t |

    travel_time is the travel time for each point of image from the trial point (zero contour)
    dx is the grid spacing
    N is the maximum travel time

    """
    grad_t_y, grad_t_x = np.gradient(travel_time, dx)

    if isinstance(travel_time, np.ma.MaskedArray):
        grad_t_y[grad_t_y.mask] = 0.0
        grad_t_y = grad_t_y.data

        grad_t_x[grad_t_x.mask] = 0.0
        grad_t_x = grad_t_x.data

    # h, w = travel_time.shape
    # coords_x, coords_y = np.arange(boundary[0], boundary[2], (boundary[2]-boundary[0])/steps), \
    #                      np.arange(boundary[1], boundary[3], (boundary[3]-boundary[1])/steps)
    coords_x, coords_y = np.linspace(boundary[0], boundary[2], steps), \
                         np.linspace(boundary[1], boundary[3], steps)

    gradx_interp = RectBivariateSpline(coords_y, coords_x, grad_t_x)
    grady_interp = RectBivariateSpline(coords_y, coords_x, grad_t_y)

    def get_velocity(position):
        """Returns normalized velocity at the position"""
        x, y = position
        vel = np.array([gradx_interp(y, x)[0][0],
                        grady_interp(y, x)[0][0]])

        return vel / np.linalg.norm(vel)

    def euler_point_update(pos, ds):
        return pos - get_velocity(pos) * ds

    def runge_kutta(pos, ds):
        """Fourth order Runge Kutta point update"""
        k1 = ds * get_velocity(pos)
        k2 = ds * get_velocity(pos - k1 / 2.0)
        k3 = ds * get_velocity(pos - k2 / 2.0)
        k4 = ds * get_velocity(pos - k3)

        return pos - (k1 + 2 * k2 + 2 * k3 + k4) / 6.0

    p = runge_kutta(starting_point, dx)

    px, py = [p[0]], [p[1]]

    for i in range(N):
        px.append(p[0])
        py.append(p[1])
        p = runge_kutta(p, dx)
    #         x = euler_point_update(x, dx)

    return px, py