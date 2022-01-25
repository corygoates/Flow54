"""A class for modeling nozzles."""

import numpy as np

from compressible_tools import *

class Nozzle:
    """A nozzle.

    Parameters
    ----------
    P0 : float
        Inlet stagnation pressure.

    T0 : float
        Inlet stagnation temperature.

    A_throat : float
        Throat area.

    A_exit : float
        Exit area.

    fluid : Species
        Fluid species passing through nozzle.
    """

    def __init__(self, P0, T0, A_throat, A_exit, fluid):
        
        # Load params
        self.P0 = P0
        self.T0 = T0
        self.A_throat = A_throat
        self.A_exit = A_exit
        self.fluid = fluid

        # Calculate derived quantities
        self.exp_ratio = A_exit/A_throat
        self.m_dot_choked = A_throat*P0*m.sqrt(self.fluid.gamma/(T0*self.fluid.R_g)*(2.0/(self.fluid.gamma+1.0))**((self.fluid.gamma+1.0)/(self.fluid.gamma-1.0)))

        # Calculate isentropic exit parameters
        self.M_exit_ise = M_of_A_A_star(A_exit/A_throat, self.fluid.gamma)
        self.T_exit_ise = T0/isentropic_temp_ratio(self.M_exit_ise, self.fluid.gamma)
        self.P_exit_ise = P0/isentropic_pressure_ratio(self.M_exit_ise, self.fluid.gamma)
        self.c_exit_ise = m.sqrt(fluid.gamma*self.fluid.R_g*self.T_exit_ise)
        self.V_exit_ise = self.M_exit_ise*self.c_exit_ise
        self.thrust_mom_ise = self.m_dot_choked*self.V_exit_ise
        self.thrust_sl_ise = self.thrust_mom_ise+A_exit*(self.P_exit_ise-101325.0)
        self.thrust_vac_ise = self.thrust_mom_ise+A_exit*self.P_exit_ise
        self.I_sp_sl_se = self.thrust_sl_ise/(self.m_dot_choked*g_0)
        self.I_sp_vac_se = self.thrust_vac_ise/(self.m_dot_choked*g_0)


    def calc_optimal_exit_area(self, p_inf):
        """Calculates the exit area which will provide optimal performance from the nozzle (i.e. p_e = p_inf).

        Parameters
        ----------
        p_inf : float
            Design ambient pressure.
        """

        # Calculate required Mach number at exit
        M = np.sqrt(2.0/(self.fluid.gamma-1.0)*((self.P0/p_inf)**((self.fluid.gamma-1.0)/self.fluid.gamma)-1.0))

        # Determine necessary expansion ratio
        area_ratio = A_A_star_of_M(M, self.fluid.gamma)

        # Calculate exit area
        A_exit = self.A_throat*area_ratio

        return A_exit, M, area_ratio


    def get_thrust(self, p_inf):
        """Calculates the thrust from the nozzle at the given ambient pressure.

        Parameters
        ----------
        p_inf : float
            Ambient pressure.

        Returns
        -------
        thrust : float
            Thrust due to momentum and pressure.
        """

        return self.thrust_mom_ise + self.A_exit*(self.P_exit_ise-p_inf)

    
    def print_info(self):

        print()
        print("Nozzle Parameters")
        print("-----------------")
        print("    Stagnation pressure: ",  self.P0, " Pa")
        print("    Stagnation temp: ",  self.T0, " K")
        print("    Ratio of heats: ", self.gamma)
        print("    Molecular weight: ", self.M_w)
        print("    Throat area: ", self.A_throat, " m^2")
        print("    Exit area: ", self.A_exit, " m^2")
        print("    Expansion ratio: ", self.exp_ratio)
        print("    Choked mass flow: ", self.m_dot_choked, " kg/s")