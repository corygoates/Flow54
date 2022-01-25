import copy

import numpy as np

from compressible_tools import *

class FuelInjector:
    """A fuel injector.

    Parameters
    ----------
    fuel : Species
        The fluid being fed through the injector.

    T : float
        Inlet temperature of injector.

    A : float
        Cross-sectional area.
    """

    def __init__(self, fuel, T, A):

        # Store
        self.fuel = fuel
        self.T = T
        self.A = A


    def calc_subcritical_injector_pressure(self, m_dot, p_ext):
        """Calculates the required pressure in the injector to supply the given massflow against the given exterior pressure. Assumes the injector is subsonic but compressible.

        Parameters
        ----------
        m_dot : float
            Required massflow.

        p_ext : float
            Exterior pressure.
        
        Returns
        -------
        p_inj : float
            Required injector pressure.
        """

        # Define function to find the root of
        def f(p_inj):
            return self.subcritical_massflow(p_inj, p_ext)-m_dot

        # Find root using secant method
        p0 = p_ext*1.1
        p1 = p_ext*1.2
        f0 = f(p0)
        f1 = f(p1)
        while abs(f1/m_dot)>1e-12:

            # Get new pressure guess
            p2 = p1-f1*(p0-p1)/(f0-f1)

            # Update for next iteration
            p0 = p1
            p1 = p2
            f0 = f1
            f1 = f(p1)

        # Check the result is subcritical
        p_crit = (0.5*(self.fuel.gamma+1.0))**(self.fuel.gamma/(self.fuel.gamma-1.0))*p_ext
        if p1 >= p_crit:
            raise RuntimeError("Subcritical assumption of injector violated. Critical pressure is {0:1.6e}".format(p_crit))

        return p1, p_crit


    def subcritical_massflow(self, p_inj, p_ext):
        """Gives the massflow through the injector based on the injector pressure assuming the flow is subcritical.

        Parameters
        ----------
        p_inj : float
            Injector pressure.

        Returns
        -------
        m_dot : float
            Massflow.
        """

        a = (p_ext/p_inj)**(2.0/self.fuel.gamma)-(p_ext/p_inj)**((self.fuel.gamma+1.0)/self.fuel.gamma)
        b = p_inj**2/(self.fuel.R_g*self.T)*a
        c = 2.0*self.fuel.gamma/(self.fuel.gamma-1.0)*b
        return self.A*np.sqrt(c)


    def calc_velocity(self, p_inj, p_ext):
        """Gives the velocity through the injector based on the injector pressure assuming the flow is subcritical.

        Parameters
        ----------
        p_inj : float
            Injector pressure.

        Returns
        -------
        V : float
            Velocity
        """

        # Calculate massflow
        m_dot = self.subcritical_massflow(p_inj, p_ext)

        # Calculate density
        rho = p_inj/(self.fuel.R_g*self.T)

        return m_dot/(rho*self.A)