from unittest import result
import scipy.optimize as sopt
import scipy.integrate as sint

from compressible_tools import *

import numpy as np
import matplotlib.pyplot as plt


def V_prime_from_M(M, gamma):
    return (2.0/((gamma-1.0)*M**2) + 1)**-0.5


def M_from_V_prime(V_prime, gamma):
    print("V'", V_prime)
    x = V_prime**-2 - 1.0
    return (0.5*(gamma-1.0)*x)**-0.5


class ConeFlow:
    """Class for calculating axisymmetric conical flows in supersonic flow."""

    def __init__(self):

        pass


    def _taylor_maccoll_state_space(self, theta, x):
        """Calculates the state-space derivatives of the Taylor-Maccoll equation.

        Parameters
        ----------
        theta : float
            Ray angle.

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
        self.gamma = gamma
        #integrator = sint.ode(self._taylor_maccoll_state_space)
        #integrator.set_integrator('vode', method='bdf', nsteps=1000000)
        #integrator.set_initial_value(np.array([V_r, V_theta]), t=B)
        #dt = -B/2000.0
        #while integrator.successful() and integrator.y[1] <= 0.0:
        #    print(integrator.t+dt, integrator.integrate(integrator.t+dt))
        theta, x = RK4(self._taylor_maccoll_state_space, np.array([V_r, V_theta]), B, 0.0, -B/2000.0)
        V_r_space = x[:,0]
        V_theta_space = x[:,1]
        plt.figure()
        plt.plot(theta, V_theta_space)
        plt.show()

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

        # This really shouldn't ever happen, technically, but it's the limiting case
        else:   
            theta_c = 0.0
            M_surf = M

        return theta_c, M_surf


    def calc_flow(self, theta_c, M, gamma):
        """Calculates the shock angle off of the cone, surface Mach number, and pressure coefficient using the Taylor-MacColl equations.
        
        Parameters
        ----------
        theta_c : float
            Cone angle in degrees.

        M : float
            Freestream Mach number.
        
        gamma : float
            Ratio of specific heats.
        
        Returns
        -------
        beta : float
            Shock angle in degrees.

        M_surf : float
            Surface Mach number.

        C_p : float
            Surface pressure coefficient.
        """

        print("Figuring out the shock angle")

        # Function to find root of
        def f(b):
            print()
            print(b)
            theta_c_guess,_ = self.cone_angle_from_taylor_maccoll(M, gamma, b)
            diff = theta_c - theta_c_guess
            print(diff)
            return diff

        # Plot function
        mach_angle = np.degrees(mu(M))
        #betas = np.linspace(mach_angle, 70.0, 100)
        #diffs = np.zeros_like(betas)
        #for i, beta in enumerate(betas):
        #    diffs[i] = f(beta)
        #plt.figure()
        #plt.plot(betas, diffs)
        #plt.show()

        # Find root
        result = sopt.root_scalar(f, bracket=[mach_angle, 70.0], method='bisect')
        beta = result.root

        # Get surface Mach number
        theta_c_guess,M_surf = self.cone_angle_from_taylor_maccoll(M, gamma, beta)

        # Get shock angle in radians
        B = np.radians(beta)

        # Calculate pressure ratio from freestream to surface
        R_P = isentropic_pressure_ratio(M, gamma)*oblique_shock_stag_pressure_ratio(M, gamma, B)/isentropic_pressure_ratio(M_surf, gamma)

        # Calculate surface pressure coefficient
        C_p = (R_P - 1.0)/(0.5*gamma*M**2)

        return beta, M_surf, C_p


    def plot_shock_to_cone_angles(self, M_range, gamma=1.4, N_machs=5, N_angles=50):
        """Plots a chart of shock angle to cone angle.
        
        Parameters
        ----------
        M_range : list
            Lower and upper bounds for the Mach number range to plot.
            
        gamma : float, optional
            Ratio of specific heats. Defaults to 1.4.

        N_machs : int, optional
            Number of Mach numbers to plot. Defaults to 5.
        
        N_angles : int, optional
            Number of shock angles to plot. Defaults to 50.
        """

        # Initialize Mach range
        Ms = np.logspace(np.log10(M_range[0]), np.log10(M_range[1]), N_machs)

        # Loop
        plt.figure()
        for i, M in enumerate(Ms):

            # Initialize range of shock angles
            mach_angle = np.degrees(mu(M))
            betas = np.linspace(mach_angle, 75.0, N_angles)

            # Loop
            theta_c = np.zeros(N_angles)
            for j, beta in enumerate(betas):

                theta_c[j],_ = cone.cone_angle_from_taylor_maccoll(M, gamma, beta)

            series = plt.plot(theta_c, betas, label=str(round(M, 2)))

            # Plot Mach angle
            plt.plot(theta_c, np.ones_like(theta_c)*mach_angle, '--', color=series[0].get_color())

        plt.legend(title='$M_\infty$')
        plt.xlabel("$\\theta_c$")
        plt.ylabel("$\\beta$")
        plt.show()



if __name__=="__main__":

    # Initialize cone
    cone = ConeFlow()

    mach_angle = np.degrees(mu(2.0))
    cone.cone_angle_from_taylor_maccoll(1.5, 1.4, mach_angle)

    ## Plot pressure coefficient as a function of Mach number
    #Ms = np.linspace(1.5, 3.0, 5)
    #Cp = np.zeros_like(Ms)
    #M_surf = np.zeros_like(Ms)
    #beta = np.zeros_like(Ms)
    #for i, M in enumerate(Ms):
    #    print()
    #    print()
    #    print("M", M)
    #    beta[i],M_surf[i],Cp[i] = cone.calc_flow(10.0, M, 1.4)

    ## Plot
    #fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(6.5, 2.2))
    #ax[0].plot(Ms, Cp)
    #ax[0].set_xlabel('$M_\infty$')
    #ax[0].set_ylabel('$C_P$')

    #ax[1].plot(Ms, beta)
    #ax[1].set_xlabel('$M_\infty$')
    #ax[1].set_ylabel('Shock Angle')

    #ax[2].plot(Ms, M_surf)
    #ax[2].set_xlabel('$M_\infty$')
    #ax[2].set_ylabel('$M_{surf}$')
    #plt.show()