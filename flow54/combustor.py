import numpy as np

from compressible_tools import *

class Combustor:
    """A ramjet combustor.

    Parameters
    ----------
    m_dot_inlet : float
        Inlet massflow.

    m_dot_inj : float
        Injector massflow.

    air : Species
        Gas species present at the inlet.

    T_inlet : float
        Inlet temperature.

    V_inlet : float
        Inlet velocity.

    p_inlet : float
        Inlet pressure.

    exhaust : Species
        Exhaust species.

    A : float
        Area.
    """

    def __init__(self, m_dot_inlet, m_dot_inj, air, T_inlet, V_inlet, p_inlet, exhaust, A):

        # Calculate exit massflow
        self.m_dot_inlet = m_dot_inlet
        self.m_dot_inj = m_dot_inj
        self.m_dot_exit = self.m_dot_inlet+self.m_dot_inj
        self.f = m_dot_inlet/m_dot_inj

        # Calculate inlet properties
        self.air = air
        self.c_inlet = np.sqrt(self.air.gamma*self.air.R_g*T_inlet)
        self.M_inlet = V_inlet/self.c_inlet
        self.T0_inlet = T_inlet*isentropic_temp_ratio(self.M_inlet, self.air.gamma)
        self.P0_inlet = p_inlet*isentropic_pressure_ratio(self.M_inlet, self.air.gamma)
        self.p_inlet = p_inlet

        # Store exahust properties
        self.exhaust = exhaust
        self.A = A

        # Calculate function of inlet properties
        a = self.M_inlet**2*(1.0+0.5*(self.air.gamma-1.0)*self.M_inlet**2)/(1.0+self.air.gamma*self.M_inlet**2)**2
        b = self.exhaust.R_g*self.air.gamma/(self.air.R_g*self.exhaust.gamma)*a
        self.g_M_inlet = ((self.f+1.0)/self.f)**2*b


    def apply_heat_load(self, delta_q):
        """Applies a heat load to the combustor.

        Parameters
        ----------
        delta_q : float
            Heat load.
        """

        # Store heat load
        self.delta_q = delta_q

        # Calculate resulting exit stagnation temperature
        self.T0_exit = (self.delta_q+self.f/(self.f+1.0)*self.air.c_p*self.T0_inlet)/self.exhaust.c_p

        # Calculate exit Mach number
        self._calc_exit_mach()

        # Calculate other exit parameters
        self._calc_exit_params()



    def _calc_exit_mach(self):
        # Calculates the exit Mach number

        # Calculate quadratic coefficients
        g_M = self.g_M_inlet*self.T0_exit/self.T0_inlet
        A = 0.5*(self.exhaust.gamma-1.0)-self.exhaust.gamma**2*g_M
        B = 1.0-2.0*self.exhaust.gamma*g_M
        C = -g_M

        # Calculate solution to quadratic formula
        w = B**2-4.0*A*C
        if abs(w)<1e-12: # Thermal choke, allowing for roundoff error
            self.M_exit = 1.0
        elif w < 0.0: # Exceeding thermal choke
            raise RuntimeError("Appied heat load has exceeded thermal choke.")
        else:

            x = np.sqrt(w)

            M_1 = np.sqrt((-B-x)/(2.0*A))
            M_2 = np.sqrt((-B+x)/(2.0*A))

            if self.M_inlet < 1.0: # Subsonic
                self.M_exit = min(M_1, M_2)
            else: # Supersonic
                self.M_exit = max(M_1, M_2)

    
    def apply_heat_load_for_choke(self):
        """Calculates the required heat load to choke the combustor. Sets properties accordingly.

        Returns
        -------
        delta_q : float
            Heat load for choke.
        """

        # Calculate necessary stagnation temperature ratio
        R_T = (1.0+self.exhaust.gamma)**2/(1.0+0.5*(self.exhaust.gamma-1.0))*self.g_M_inlet

        # Calculate necessary exit stagnation temperature
        self.T0_exit = self.T0_inlet/R_T

        # Calculate heat load
        self.delta_q = self.exhaust.c_p*self.T0_exit - self.f/(self.f+1.0)*self.air.c_p*self.T0_inlet

        # Set other parameters
        self._calc_exit_mach()
        self._calc_exit_params()

        return self.delta_q


    def _calc_exit_params(self):
        # Calculates resulting exit static pressure and temperature and stagnation pressure

        # Exit temperature
        self.T_exit = self.T0_exit/isentropic_temp_ratio(self.M_exit, self.exhaust.gamma)

        # Exit pressure (from conservation of momentum)
        self.p_exit = self.p_inlet*(1.0+self.air.gamma*self.M_inlet**2)/(1.0+self.exhaust.gamma*self.M_exit**2)

        # Exit stagnation pressure
        self.P0_exit = self.p_exit*isentropic_pressure_ratio(self.M_exit, self.exhaust.gamma)