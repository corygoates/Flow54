import numpy as np
from compressible_tools import *
import matplotlib.pyplot as plt


def cone_angle_from_taylor_maccoll(M, gamma, beta):
    """Calculates the cone angle which will support a conical shock with the given upstream Mach number and shock angle.
    
    Parameters
    ----------
    M : float
        Upstream Mach number.
        
    beta : float
        Shock angle in degrees.
    
    gamma : float
        Ratio of specific heats."""

    # Get shock angle in radians
    B = np.radians(beta)

    # Calculate aft deflection
    d = oblique_wedge_angle(M, gamma, B)
    
    # Calculate aft Mach number
    M2 = oblique_shock_aft_M(M, gamma, B, d)

    # Get aft V'
    V_prime = V_prime_from_M(M2, gamma)

    # Calculate V_r and V_theta
    a = B-d
    V_r = V_prime*np.cos(a)
    V_theta = -V_prime*np.sin(a)
    
    # Integrate
    theta, x = RK4(taylor_maccoll_state_space, np.array([V_r, V_theta]), B, 0.0, -B/1000.0)
    V_r_space = x[:,0]
    V_theta_space = x[:,1]

    # Find where we cross V_theta = 0
    for i in range(len(V_theta_space)-1):

        if np.sign(V_theta_space[i]) != np.sign(V_theta_space[i+1]):

            theta_c = theta[i+1] - (V_theta_space[i+1] * (theta[i]-theta[i+1])) / (V_theta_space[i] - V_theta_space[i+1])
            return theta_c

    return np.nan


def taylor_maccoll_state_space(theta, x, gamma=1.4):
    """Calculates the state-space derivatives of the Taylor-Maccoll equation.
    
    Parameters
    ----------
    theta : float
    
    x : ndarray
        First component is V_r; second is V_theta.

    Returns
    -------
    x_dot : ndarray
    """

    # Initialize
    x_dot = np.zeros_like(x)
    x1 = x[0]
    x2 = x[1]

    # dV_r/dtheta
    x_dot[0] = x2

    # d2V_r/dtheta2
    f = (gamma-1.0)*0.5*(1.0 - x1**2 - x2**2)
    x_dot[1] = (x1*x2 - f*(2.0*x1 + x2/np.tan(theta))) / (f - x2**2)

    return x_dot


def V_prime_from_M(M, gamma):
    return (2.0/((gamma-1)*M**2) + 1)**-0.5


class Cone:
    """Class for calculating axisymmetric conical flows in supersonic flow.
    
    Parameters
    ----------
    theta : float
        Cone half angle in degrees.
    """

    def __init__(self, theta):

        self.theta = np.radians(theta)


    def solve_taylor_maccoll(self, M, gamma):
        """Solves the Taylor-Maccoll equation over the cone.
        
        Parameters
        ----------
        M : float
            Freestream Mach number.

        gamma : float
            Ratio of specific heats.
        """

        # Initial guess for shock angle will be the Mach angle
        beta0 = np.arcsin(1.0/M)


if __name__=="__main__":

    # Declare range of Mach numbers and shock angles
    Ms = [1.5, 2.0, 3.0, 5.0]
    N_betas = 50

    # Loop
    plt.figure()
    for i, M in enumerate(Ms):

        # Determine minimum shock angle
        beta_min = np.degrees(np.arcsin(1.0/M))
        betas = np.linspace(beta_min, 65.0, N_betas)

        # Loop
        theta_c = np.zeros(N_betas)
        for j, beta in enumerate(betas):

            theta_c[j] = np.degrees(cone_angle_from_taylor_maccoll(M, 1.4, beta))

        plt.plot(theta_c, betas, label=str(M))

    plt.legend()
    plt.xlabel("$\\theta_c$")
    plt.ylabel("$\\beta$")
    plt.show()