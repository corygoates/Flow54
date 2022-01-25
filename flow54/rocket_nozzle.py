"""A class for modeling rocket nozzles."""

import numpy as np

from compressible_tools import *

class Nozzle:
    """A rocket nozzle.

    Parameters
    ----------
    P0 : float
        Chamber stagnation pressure.

    T0 : float
        Chamber stagnation temperature.

    A_throat : float
        Throat area.

    A_exit : float
        Exit area.

    gamma : float
        Species ratio of specific heats.

    M_w : float
        Species molecular weight.
    """

    def __init__(self, P0, T0, A_throat, A_exit, gamma, M_w):
        
        # Load params
        self.P0 = P0
        self.T0 = T0
        self.A_throat = A_throat
        self.A_exit = A_exit
        self.gamma = gamma
        self.M_w = M_w

        # Calculate derived quantities
        self.R_g = R_u/M_w
        self.exp_ratio = A_exit/A_throat
        self.m_dot_choked = A_throat*P0*m.sqrt(gamma/(T0*self.R_g)*(2.0/(gamma+1.0))**((gamma+1.0)/(gamma-1.0)))

        # Calculate isentropic exit parameters
        self.M_exit_ise = M_of_A_A_star(A_exit/A_throat, gamma)
        self.T_exit_ise = T0/isentropic_temp_ratio(self.M_exit_ise, gamma)
        self.P_exit_ise = P0/isentropic_pressure_ratio(self.M_exit_ise, gamma)
        self.c_exit_ise = m.sqrt(gamma*self.R_g*self.T_exit_ise)
        self.V_exit_ise = self.M_exit_ise*self.c_exit_ise
        self.thrust_mom_ise = self.m_dot_choked*self.V_exit_ise
        self.thrust_sl_ise = self.thrust_mom_ise+A_exit*(self.P_exit_ise-101325.0)
        self.thrust_vac_ise = self.thrust_mom_ise+A_exit*self.P_exit_ise
        self.I_sp_sl_se = self.thrust_sl_ise/(self.m_dot_choked*g_0)
        self.I_sp_vac_se = self.thrust_vac_ise/(self.m_dot_choked*g_0)

    
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