from unittest import result
import scipy.optimize as sopt

from compressible_tools import *

import numpy as np
import matplotlib.pyplot as plt


def V_prime_from_M(M, gamma):
    return (2.0/((gamma-1.0)*M**2) + 1)**-0.5


def M_from_V_prime(V_prime, gamma):
    return np.sqrt(2.0/((gamma-1.0)*(V_prime**-2 - 1.0)))


class Cone:
    """Class for calculating axisymmetric conical flows in supersonic flow.
    
    Parameters
    ----------
    theta : float
        Cone half angle in degrees.
    """

    def __init__(self, theta):

        self.theta = np.radians(theta)


    def _taylor_maccoll_state_space(self, theta, x):
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
        x22 = x2*x2
        f = 0.5*(self.gamma-1.0)*(1.0 - x1**2 - x22)
        denom = f - x22
        x_dot[1] = (x1*x22 - f*(2.0*x1 + x2/np.tan(theta))) / denom

        return x_dot


    def cone_angle_from_taylor_maccoll(self, M, gamma, beta):
        """Calculates the cone angle which will support a conical shock with the given upstream Mach number and shock angle.

        Parameters
        ----------
        M : float
            Upstream Mach number.

        beta : float
            Shock angle in degrees.

        gamma : float
            Ratio of specific heats.

        Returns
        -------
        theta_c : float
            Cone angle in degrees which will support the given shock angle.

        M_surf : float
            Surface Mach number.
        """

        self.gamma = gamma

        # Get shock angle in radians
        B = np.radians(beta)

        # Calculate aft deflection
        d = oblique_wedge_angle(M, gamma, B)

        # Calculate aft Mach number
        M2 = oblique_shock_aft_M(M, gamma, B, d)

        # Get aft V'
        V_prime = V_prime_from_M(M2, gamma)

        # Calculate V_r and V_theta
        a = B - d
        V_r = V_prime*np.cos(a)
        V_theta = -V_prime*np.sin(a)

        # Integrate
        theta, x = RK4(self._taylor_maccoll_state_space, np.array([V_r, V_theta]), B, 0.0, -B/1000.0)
        V_r_space = x[:,0]
        V_theta_space = x[:,1]
        #plt.figure()
        #plt.plot(np.degrees(theta), V_r_space, label="$V'_r$")
        #plt.plot(np.degrees(theta), V_theta_space, label="$V'_\\theta$")
        #plt.xlabel("$\\theta$")
        #plt.ylabel("$V'$")
        #plt.legend()
        #plt.show()

        # Find where we cross V_theta = 0
        for i in range(len(V_theta_space)-1):

            if abs(V_theta_space[i]) < 1e-12:
                theta_c = np.degrees(theta[i])
                V_r_c = np.degrees(V_r_space[i])
                M_surf = M_from_V_prime(V_r_c, gamma)
                break

            elif np.sign(V_theta_space[i]) != np.sign(V_theta_space[i+1]):
                d_theta = theta[i] - theta[i+1]
                theta_c = np.degrees(theta[i+1] - (V_theta_space[i+1] * d_theta) / (V_theta_space[i] - V_theta_space[i+1]))
                V_r_c = V_r_space[i]*(theta[i] - theta_c)/d_theta + V_r_space[i+1]*(theta_c - theta[i+1])/d_theta
                M_surf = M_from_V_prime(V_r_c, gamma)
                break

        else:   
            theta_c = np.nan
            M_surf = np.nan

        return theta_c, M_surf


    def shock_angle_from_taylor_maccoll(self, M, gamma):
        """Calculates the shock angle off of the cone using the Taylor-MacColl equations.
        
        Parameters
        ----------
        M : float
            Freestream Mach number.
        
        gamma : float
            Ratio of specific heats.
        
        Returns
        -------
        beta : float
            Shock angle in degrees.

        C_p : float
            Pressure coefficient on surface of cone.
        """

        # Function to find root of
        def f(b):
            theta_c,_ = self.cone_angle_from_taylor_maccoll(M, gamma, b)
            diff = np.degrees(self.theta) - theta_c
            return diff

        # Find root
        mach_angle = np.degrees(mu(M))
        result = sopt.root_scalar(f, bracket=[mach_angle, 65.0])
        #result = sopt.root_scalar(f, x0=mach_angle*1.05, x1=mach_angle*1.2)
        beta = result.root
        _,M_surf = self.cone_angle_from_taylor_maccoll(M, gamma, beta)

        return beta

    
    def calc_surface_properties(self, M, gamma):
        """Calculates the surface Mach number and pressure on the cone.
        
        Parameters
        ----------
        M : float
            Freestream Mach number.
            
        gamma : float
            Ratio of specific heats.
        """

        pass


if __name__=="__main__":

    cone = Cone(10.0)

    # Declare range of Mach numbers and shock angles
    Ms = [1.5, 2.0, 3.0, 5.0]
    N_betas = 200

    print(cone.shock_angle_from_taylor_maccoll(1.5, 1.4))
    print(cone.shock_angle_from_taylor_maccoll(2.0, 1.4))
    print(cone.shock_angle_from_taylor_maccoll(3.0, 1.4))
    print(cone.shock_angle_from_taylor_maccoll(5.0, 1.4))

    # Loop
    plt.figure()
    for i, M in enumerate(Ms):

        # Initialize range of shock angles
        mach_angle = np.degrees(mu(M))
        betas = np.linspace(mach_angle, 75.0, N_betas)

        # Loop
        theta_c = np.zeros(N_betas)
        for j, beta in enumerate(betas):

            theta_c[j],_ = cone.cone_angle_from_taylor_maccoll(M, 1.4, beta)

        series = plt.plot(theta_c, betas, label=str(M))

        # Plot Mach angle
        plt.plot(theta_c, np.ones_like(theta_c)*mach_angle, '--', color=series[0].get_color())

    plt.legend(title='$M_\infty$')
    plt.xlabel("$\\theta_c$")
    plt.ylabel("$\\beta$")
    plt.show()