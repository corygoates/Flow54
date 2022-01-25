import copy
import numpy as np

from flow54.compressible_tools import isentropic_pressure_ratio, oblique_shock_angle, oblique_shock_aft_M, expansion_fan_aft_M, oblique_shock_pressure_ratio, oblique_shock_stag_pressure_ratio

class DiamondAirfoil:
    """Defines a supersonic, diamond airfoil. Analyzed using shock-expansion theory.

    Parameters
    ----------
    theta : float
        Wedge half-angle given in degrees. Must be nonzero.

    c : float
        Chord length.
    """

    def __init__(self, theta, c):

        # Store
        self._theta = np.radians(theta)
        self._c = c
        self._t = self._c*np.tan(self._theta)


    def set_state(self, M, alpha, gamma, p_inf, T_inf, c_inf, rho_inf, mu_inf, c_p, Pr):
        """Sets the state of the airfoil.

        Parameters
        ----------
        M : float
            Freestream Mach number.

        alpha : float
            Angle of attack in degrees.

        gamma : float
            Ratio of specific heats.

        p_inf : float
            Freestream static pressure.

        T_inf : float
            Freestream temperature.

        c_inf : float
            Freestream speed of sound.

        rho_inf : float
            Freestream density.
        
        mu_inf : float
            Freestream dynamic viscosity.

        c_p : float
            Specific heat at constant pressure.

        Pr : float
            Prandtl number
        """

        # Store
        self._M = M
        self._alpha = np.radians(alpha)
        self._gamma = gamma
        self._p_inf = p_inf
        self._T_inf = T_inf
        self._c_inf = c_inf
        self._rho_inf = rho_inf
        self._mu_inf = mu_inf
        self._c_p = c_p
        self._Pr = Pr

        # Calculate derived properties
        self._P0_inf = isentropic_pressure_ratio(self._M, self._gamma)*self._p_inf
        self._V_inf = self._M*self._c_inf
        self._Re_inf = self._rho_inf*self._V_inf*self._c*np.cos(self._alpha)/(self._mu_inf*np.cos(self._theta))

        # Calculate maximum shock angle
        a = np.sqrt(16.0*(self._gamma+1.0)+8.0*self._M**2*(self._gamma**2-1)+self._M**4*(self._gamma+1.0)**2)
        b = self._M**2*(self._gamma-1.0)+4.0
        B_max = 0.5*np.arccos((b-a)/(2.0*self._gamma*self._M**2))

        # Calculate corresponding maximum turning angle
        a = 2.0*(self._M**2*np.sin(B_max)**2-1.0)
        b = np.tan(B_max)*(2.0+self._M**2*(self._gamma+np.cos(2.0*B_max)))
        theta_max = np.arctan(a/b)

        # Check angle
        if theta_max < abs(self._alpha)+self._theta:
            raise RuntimeError("Airfoil experiences subsonic flow aft of the bow shock at a Mach number of {0} and angle of attack of {1} degrees.".format(self._M, np.degrees(self._alpha)))


    def get_alpha_max(self, M, gamma):
        """Calculates the maximum angle of attack for this airfoil based on the appearance of subsonic flow.

        Parameters
        ----------
        M : float
            Freestream Mach number.

        gamma : float
            Ratio of specific heats.

        Returns
        -------
        alpha_max : float
            Maximum angle of attack in degrees.
        """

        ## Calculate maximum shock angle (based on detached shock)
        #a = np.sqrt(16.0*(gamma+1.0)+8.0*M**2*(gamma**2-1)+M**4*(gamma+1.0)**2)
        #b = M**2*(gamma-1.0)+4.0
        #B_max = 0.5*np.arccos((b-a)/(2.0*gamma*M**2))

        # Calculate maximum shock angle (based on subsonic flow)
        a = np.sqrt(16.0*gamma*M**2+((gamma-3.0)*M**2+(gamma+1.0)*M**4)**2)
        b = (gamma-3.0)*M**2+(gamma+1.0)*M**4+a
        B_max = np.arcsin(np.sqrt(b/(4.0*gamma*M**4)))

        # Calculate corresponding maximum turning angle
        a = 2.0*(M**2*np.sin(B_max)**2-1.0)
        b = np.tan(B_max)*(2.0+M**2*(gamma+np.cos(2.0*B_max)))
        theta_max = np.arctan(a/b)

        # Calculate max angle of attack
        alpha_max = theta_max-self._theta
        return np.degrees(alpha_max)
        

    def get_mach_numbers(self):
        """Calculates the Mach numbers on each face of the airfoil.

                3   5      
        ---->    <>                           
                2   4   

        Returns
        -------
        M2, M3, M4, M5 : float
            Mach numbers.
        """
        
        # M2
        turn_angle = self._alpha+self._theta
        if turn_angle > 0.0:

            # Calculate shock angle
            beta2 = oblique_shock_angle(self._M, self._gamma, turn_angle)

            # Get Mach number
            M2 = oblique_shock_aft_M(self._M, self._gamma, beta2, turn_angle)

        elif turn_angle < 0.0:

            # Calculate aft Mach number
            M2 = expansion_fan_aft_M(self._M, self._gamma, -turn_angle)

        else:

            # No change
            M2 = self._M
        
        # M3
        turn_angle = self._theta-self._alpha
        if turn_angle > 0.0:

            # Calculate shock angle
            beta3 = oblique_shock_angle(self._M, self._gamma, turn_angle)

            # Get Mach number
            M3 = oblique_shock_aft_M(self._M, self._gamma, beta3, turn_angle)

        elif turn_angle < 0.0:

            # Calculate aft Mach number
            M3 = expansion_fan_aft_M(self._M, self._gamma, -turn_angle)

        else:

            # No change
            M3 = self._M

        # p4; always an expansion fan
        turn_angle = 2.0*self._theta

        # Calculate aft Mach number
        M4 = expansion_fan_aft_M(M2, self._gamma, turn_angle)

        # p5; always an expansion fan

        # Calculate aft Mach number
        M5 = expansion_fan_aft_M(M3, self._gamma, turn_angle)

        return M2, M3, M4, M5
        

    def get_pressures(self):
        """Calculates the static pressures on each face of the airfoil.

                3   5      
        ---->    <>                           
                2   4   

        Returns
        -------
        p2, p3, p4, p5 : float
            Static pressures.
        """

        # Get Mach numbers
        M2, M3, M4, M5 = self.get_mach_numbers()
        
        # p2
        turn_angle = self._alpha+self._theta
        if turn_angle > 0.0:

            # Calculate shock angle
            beta2 = oblique_shock_angle(self._M, self._gamma, turn_angle)

            # Get pressure
            p2 = oblique_shock_pressure_ratio(self._M, self._gamma, beta2)*self._p_inf

        elif turn_angle < 0.0:

            # Calculate pressure
            p2 = self._P0_inf/isentropic_pressure_ratio(M2, self._gamma)

        else:

            # No change
            p2 = self._p_inf
        
        # p3
        turn_angle = self._theta-self._alpha
        if turn_angle > 0.0:

            # Calculate shock angle
            beta3 = oblique_shock_angle(self._M, self._gamma, turn_angle)

            # Get pressure
            p3 = oblique_shock_pressure_ratio(self._M, self._gamma, beta3)*self._p_inf

        elif turn_angle < 0.0:

            # Calculate pressure
            p3 = self._P0_inf/isentropic_pressure_ratio(M3, self._gamma)

        else:

            # No change
            p3 = self._p_inf

        # p4; always an expansion fan
        turn_angle = 2.0*self._theta

        # Calculate pressure
        p4 = p2*isentropic_pressure_ratio(M2, self._gamma)/isentropic_pressure_ratio(M4, self._gamma)

        # p5; always an expansion fan

        # Calculate pressure
        p5 = p3*isentropic_pressure_ratio(M3, self._gamma)/isentropic_pressure_ratio(M5, self._gamma)

        return p2, p3, p4, p5


    def get_inviscid_lift_and_drag(self):
        """Calculates the lift and drag."""

        # Calculate pressures
        p2, p3, p4, p5 = self.get_pressures()

        # Calculate axial and normal forces
        N = 0.5*(p2+p4-p3-p5)*self._c
        A = 0.5*(p2+p3-p4-p5)*self._t

        # Calculate lift and drag
        D = A*np.cos(self._alpha)+N*np.sin(self._alpha)
        L = N*np.cos(self._alpha)-A*np.sin(self._alpha)

        return L, D

    
    def get_inviscid_lift_and_drag_coefs(self):
        """Calculates the lift and drag coefficients."""

        # Get lift and drag forces
        L, D = self.get_inviscid_lift_and_drag()

        # Nondimensionalize
        CL = 2.0*L/(self._c*self._gamma*self._M**2*self._p_inf)
        CD = 2.0*D/(self._c*self._gamma*self._M**2*self._p_inf)

        return CL, CD


    def get_viscous_lift_and_drag(self, compressible=True, force_1_7=False):
        """Calculates the viscous lift and drag (includes flat plate model).

        Parameters
        ----------
        compressible : bool, opt
            Whether compressibility should be accounted for in the boundary layer model. Defaults to True.

        force_1_7 : bool, opt
            Whether to force use of the 1/7th power law correlation. Defaults to False.

        """

        # Get inviscid lift and drag
        L, D = self.get_inviscid_lift_and_drag()

        # Calculate incompressible skin friction coefficient

        # Turbulent
        if self._Re_inf > 1.0e7 or force_1_7:
            C_f = (7.0/255.0)/self._Re_inf**(1.0/7.0)

        # Transitional
        elif self._Re_inf <= 1.0e7 and self._Re_inf > 500000.0:
            C_f = -3.116/self._Re_inf**0.5 + 0.04096/self._Re_inf**(1.0/7.0)

        # Laminar
        else:
            C_f = 1.328/self._Re_inf**0.5

        # Apply correction
        if compressible:

            # Calculate correction

            # 1/7th power law
            if self._Re_inf > 500000.0 or force_1_7:
                Rf = self._Pr**(1.0/3.0)
                T_avg = self._T_inf+(Rf-7.0/9.0)*self._V_inf**2/(2.0*self._c_p)
                corr = ((self._T_inf/T_avg)**2.5*((T_avg+120.0)/(self._T_inf+120.0)))**(1.0/7.0)

            # Laminar
            else:
                Rf = np.sqrt(self._Pr)
                T_avg = self._T_inf+(Rf-8.0/15.0)*self._V_inf**2/(2.0*self._c_p)
                corr = ((self._T_inf/T_avg)**2.5*((T_avg+120.0)/(self._T_inf+120.0)))**0.5

            # Apply correction
            C_f /= corr

        # Redimensionalize and multiply by 2 to account for both sides of the plate
        F_visc = 2.0*C_f/np.cos(self._theta)*(0.5*self._c*self._gamma*self._M**2*self._p_inf)

        # Apply to lift and drag
        L -= F_visc*np.sin(self._alpha)
        D += F_visc*np.cos(self._alpha)

        return L, D

    
    def get_viscous_lift_and_drag_coefs(self, compressible=True, force_1_7=False):
        """Calculates the lift and drag coefficients.

        Parameters
        ----------
        compressible : bool, opt
            Whether compressibility should be accounted for in the boundary layer model. Defaults to True.

        force_1_7 : bool, opt
            Whether to force use of the 1/7th power law correlation. Defaults to False.

        """

        # Get lift and drag forces
        L, D = self.get_viscous_lift_and_drag(compressible=compressible, force_1_7=force_1_7)

        # Nondimensionalize
        CL = 2.0*L/(self._c*self._gamma*self._M**2*self._p_inf)
        CD = 2.0*D/(self._c*self._gamma*self._M**2*self._p_inf)

        return CL, CD