import numpy as np

from compressible_tools import *

class Diffuser:
    """A ramjet diffuser. Assumed isentropic with choked throat.

    Parameters
    ----------
    T_inf : float
        Freestream temperature.

    p_inf : float
        Freestream pressure.

    V_inf : float
        Freestream velocity.

    gas : Species
        Instance of Species describing the gas in the diffuser.

    d_inlet : float
        Diameter of diffuser inlet.

    d_exit : float
        Diameter of diffuser exit.
    """

    def __init__(self, T_inf, p_inf, V_inf, gas, d_inlet, d_exit):

        # Store
        self.T_inf = T_inf
        self.p_inf = p_inf
        self.V_inf = V_inf
        self.gas = gas
        self.d_inlet = d_inlet
        self.d_exit = d_exit

        # Calculate inlet parameters
        self.c_inf = np.sqrt(self.gas.gamma*self.gas.R_g*self.T_inf)
        self.M_inf = self.V_inf/self.c_inf

        # Calculate geometry
        self.A_inlet = 0.25*np.pi*self.d_inlet**2
        self.A_exit = 0.25*np.pi*self.d_exit**2

        # Calculate stagnation properties
        self.T0 = self.T_inf*isentropic_temp_ratio(self.M_inf, self.gas.gamma)
        self.P0 = self.p_inf*isentropic_pressure_ratio(self.M_inf, self.gas.gamma)

        # Calculate throat temperature
        self.T_throat = self.T0/isentropic_temp_ratio(1.0, self.gas.gamma)

        # Calculate sonic area
        area_ratio = A_A_star_of_M(self.M_inf, self.gas.gamma)
        self.A_star = self.A_inlet/area_ratio

        # Calculate exit Mach number
        self.M_exit = M_of_A_A_star(self.A_exit/self.A_star, self.gas.gamma, subsonic=True)

        # Calculate exit temperature and pressure
        self.T_exit = self.T0/isentropic_temp_ratio(self.M_exit, self.gas.gamma)
        self.p_exit = self.P0/isentropic_pressure_ratio(self.M_exit, self.gas.gamma)

        # Calculate exit velocity
        self.c_exit = np.sqrt(self.gas.gamma*self.gas.R_g*self.T_exit)
        self.V_exit = self.M_exit*self.c_exit

        # Calculate diffuser massflow
        self.m_dot = self.A_star*self.P0*np.sqrt(self.gas.gamma/(self.T0*self.gas.R_g)*(2.0/(self.gas.gamma+1.0))**((self.gas.gamma+1.0)/(self.gas.gamma-1.0)))