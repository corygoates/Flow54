import copy

import math as m
import numpy as np


# Constants
g_0 = 9.81
R_u = 8314.462618 # J/kmol K


def M_star_of_M(M, gamma):
    # Calculates M* as a function of M and gamma

    return np.sqrt(((gamma+1)*M**2)/(1+(gamma-1)*M**2))


def A_A_star_of_M(M, *args):
    # Calculates A/A* as a function of M and gamma

    gamma = args[0]
    return 1.0/M*(2.0/(gamma+1.0)*(1.0+0.5*(gamma-1.0)*M**2))**((gamma+1.0)/(2.0*(gamma-1.0)))


def dA_A_star_of_M(M, *args):
    """Calculates d(A/A*)/dM as a function of M and gamma"""

    gamma = args[0]
    a = 1.0+0.5*(gamma-1.0)*M**2
    b = (a/(gamma+1.0))**((gamma+1.0)/(2.0*(gamma-1.0)))
    c = (M**2-1.0)/(M**2*(2.0+M**2*(gamma-1.0)))*b
    return 2.0**((1.0-3.0*gamma)/(2.0-2.0*gamma))*c


def newtons_method(f, x0, args=(), dfdx=None, eps=1.0e-5, max_iter=1000, fe=None):
    # Uses Newton's method to calculate the root of f using initial guess x0

    # Initialize loop
    iteration = 0
    e = 2.0*eps
    x = copy.copy(x0)

    # Loop
    while e > eps and iteration < max_iter:

        # Get necessary values
        fi = f(x, *args)
        dfdxi = dfdx(x, *args)

        # Calculate update
        x -= fi/dfdxi

        # Estimate error
        if fe is None:
            e = abs(fi*x/dfdxi)
        else:
            e = fe(x, *args)

        # Update iteration
        iteration += 1

    return x


def M_of_A_A_star(A_A_star, gamma, subsonic=False):
    """Uses Newton's method to calculate M as a function of A/A* and gamma."""

    # Define root function
    def f(M, *args):
        return A_A_star_of_M(M, *args)-A_A_star

    # Define error funciton
    def fe(M, *args):
        return abs((A_A_star_of_M(M, *args)-A_A_star)/A_A_star)

    # Subsonic starting condition
    if subsonic:
        M0 = 0.01
    
    # Supersonic
    else:
        M0 = 10.0

    # Get Mach number
    M = newtons_method(f, M0, dfdx=dA_A_star_of_M, args=(gamma,), fe=fe)

    return M


def normal_shock_pressure_ratio(M, gamma):
    """Gives the normal shock static pressure ratio as a function of upstream Mach number."""

    return 1.0+2.0*gamma/(gamma+1.0)*(M**2.0-1.0)


def normal_shock_temp_ratio(M, gamma):
    """Gives the normal shock static temperature ratio as a function of upstream Mach number."""

    return normal_shock_pressure_ratio(M, gamma)/normal_shock_density_ratio(M, gamma)


def normal_shock_stag_pressure_ratio(M, *args):
    """Gives the normal shock stagnation pressure ratio as a function of upstream Mach number."""

    gamma = args[0]
    a = (0.5*(gamma+1.0)*M)**2.0
    b = a/(1.0+0.5*(gamma-1.0)*M**2.0)
    c = b**(gamma/(gamma-1.0))
    d = 2.0/((gamma+1.0)*(gamma*M**2-0.5*(gamma-1.0))**(1/(gamma-1.0)))*c

    return d


def normal_shock_aft_mach(M, gamma):
    """Gives the normal shock aft Mach number as a function of upstream Mach number."""

    return m.sqrt((1.0+0.5*(gamma-1.0)*M**2.0)/(gamma*M**2.0-0.5*(gamma-1.0)))


def normal_shock_density_ratio(M, gamma):
    """GIves the normal shock density ratio as a function of upstream Mach number."""

    return ((gamma+1.0)*M**2.0)/(2.0+(gamma-1.0)*M**2.0)


def dnormal_shock_stag_pressure_ratio(M, *args):
    """Calculates d(P02/P01)/dM as a function of M."""

    gamma = args[0]
    a = 2.0**(3.0-2.0*gamma/(gamma-1.0))*gamma*(M**2.0-1.0)**2.0
    b = a*(((gamma+1.0)*M)**2.0/(1.0+0.5*(gamma-1.0)*M**2.0))**(gamma/(gamma-1.0))
    c = b*(0.5+gamma*(M**2.0-0.5))**(-1.0/(gamma-1.0))
    d = -c/((gamma+1.0)*M*(2.0+M**2.0*(gamma-1.0))*(1.0+gamma*(2.0*M**2.0-1.0)))

    return d


def M1_of_stag_pressure_ratio(stag_pressure_ratio, gamma):
    """Uses Newton's method to calculate M as a function of P02/P01 and gamma."""

    # Define root function
    def f(M, *args):
        return normal_shock_stag_pressure_ratio(M, *args)-stag_pressure_ratio

    # Define error funciton
    def fe(M, *args):
        return abs((normal_shock_stag_pressure_ratio(M, *args)-stag_pressure_ratio)/stag_pressure_ratio)
    
    # Initial guess (always supersonic)
    M0 = 3.0

    # Get Mach number
    M = newtons_method(f, M0, dfdx=dnormal_shock_stag_pressure_ratio, args=(gamma,), fe=fe)

    return M


def isentropic_temp_ratio(M, gamma):
    """Gives the ratio T0/T as a function of M."""

    return 1.0+0.5*(gamma-1.0)*M**2.0


def isentropic_pressure_ratio(M, gamma):
    """Gives the ratio P0/P as a function of M."""

    return isentropic_temp_ratio(M, gamma)**(gamma/(gamma-1.0))


def rayleigh_pitot(M, *args):
    """Calculates the pressure ratio given by the supersonic Rayleigh-Pitot equation."""

    gamma = args[0]

    return ((gamma+1.0)/2.0*M**2)**(gamma/(gamma-1.0))/(2.0*gamma/(gamma+1.0)*M**2-(gamma-1.0)/(gamma+1.0))**(1/(gamma-1.0))


def M_from_rayleigh_pitot(P_ratio, gamma):
    """Calculates the freestream Mach number as a function of the pressure ratio P_0/p_inf."""

    # Determine subsonic Mach number
    M_sub =  np.sqrt(2./(gamma-1.0)*(P_ratio**((gamma-1.0)/gamma)-1.0))

    # Check validity of subsonic solution
    if M_sub <= 1.0:
        return M_sub
    
    # Determine supersonic Mach number using Newton's method
    else:

        # Function to find root of
        def f(M, *args):
            gamma = args[0]
            return rayleigh_pitot(M, gamma)-P_ratio

        # Derivative
        def df_dM(M, *args):
            gamma = args[0]
            return gamma*M*(2.0*M**2-1.0)*(M**2*(gamma+1.0)/2.0)**(1.0/(gamma-1.0))/(2.0*gamma/(gamma+1.0)*M**2-(gamma-1.0)/(gamma+1.0))**(gamma/(gamma-1.0))

        # Error function
        def fe(M, *args):
            gamma = args[0]
            return abs(f(M, *args)/P_ratio)

        # Initial guess
        M0 = 3.0

        # Get result from Newton's method
        M_super = newtons_method(f, M0, dfdx=df_dM, args=(gamma,), fe=fe)
        return M_super


def oblique_wedge_angle(M, gamma, beta):
    # Gives the oblique wedge angle as a function of upstream Mach number, gamma, and shock angle

    Mn12 = (M*np.sin(beta))**2
    a = 2.*(Mn12-1.)
    b = np.tan(beta)*(2.+M**2*(gamma+np.cos(2.*beta)))
    return np.arctan(a/b)


def mu(M):
    # Calculates the Mach angle

    return m.asin(1./M)


def oblique_shock_angle_iterative(M, gamma, theta, weak_solution=True):
    # Gives the oblique shock angle as a function of the upstream Mach number, gamma, and shock angle
    # Uses Whitmore's iterative method

    # Calculate coefficients
    a = (1.+0.5*(gamma-1.)*M**2)*np.tan(theta)
    b = M**2-1.
    c = (1.+0.5*(gamma+1.)*M**2)*np.tan(theta)

    # Initialize
    if weak_solution:
        step = np.radians(1.)
        x0 = np.tan(3.*mu(M)+step)
    else:
        x0 = np.tan(np.pi/2.1) # Just below normal
    
    # Steps up initial guess for weak solution until a realistic result is obtained
    while True:

        # Iterate
        while True:

            # Update
            if weak_solution: # Weak solution needs some relaxation
                x1 = x0 - 0.1*(a*x0**3-b*x0**2+c*x0+1.)/(3.*a*x0**2-2.*b*x0+c)
            else:
                x1 = x0 - (a*x0**3-b*x0**2+c*x0+1.)/(3.*a*x0**2-2.*b*x0+c)

            # Check error
            err = abs((x1-x0)/x1)

            # Prepare for next iteration
            x0 = x1

            # Exit
            if err < 1e-12:
                break
            
        # Check for realistic angle
        if x0 > 0.:
            break
        else:

            # Update initial guess and try again
            step += np.radians(1.)
            x0 = np.tan(mu(M)+step)

    return np.arctan(x0)


def oblique_shock_angle(M, gamma, theta, weak_solution=True):
    # Calculates the oblique shock angle as a function of Mach number, gamma, and wedge angle

    # Calculate prereqs
    a = 1+0.5*(gamma-1.)*M**2
    l = np.sqrt((M**2-1.)**2-3.*a*(1.+0.5*(gamma+1.)*M**2)*np.tan(theta)**2)
    xi = ((M**2-1.)**3 - 9.*a*(a+0.25*(gamma+1.)*M**4)*np.tan(theta)**2)/l**3
    d = int(weak_solution)

    # Calculate angle
    b = M**2-1.
    c = b + 2.*l*np.cos((4.*np.pi*d+np.arccos(xi))/3.)
    d = c/(3.*a*np.tan(theta))

    return np.arctan(d)


def oblique_shock_pressure_ratio(M, gamma, beta):
    # Calculates the static pressure ratio across the given shock

    return 1. + 2.*gamma/(gamma+1.)*((M*np.sin(beta))**2-1.)


def oblique_shock_density_ratio(M, gamma, beta):
    # Calculates the density ratio across the given shock

    return (gamma+1.)*(M*np.sin(beta))**2/(2.+(gamma-1.)*(M*np.sin(beta))**2)


def oblique_shock_temperature_ratio(M, gamma, beta):
    # Calculates the temperature ratio across the given shock

    return oblique_shock_pressure_ratio(M, gamma, beta)/oblique_shock_density_ratio(M, gamma, beta)


def oblique_shock_aft_M(M, gamma, beta, theta):
    # Calculates the Mach number aft of the given shock

    Mn1 = M*np.sin(beta)
    Mn12 = Mn1*Mn1
    Mn2 = np.sqrt((1.+0.5*(gamma-1.)*Mn12)/(gamma*Mn12-0.5*(gamma-1.)))
    return Mn2/np.sin(beta-theta)


def oblique_shock_stag_pressure_ratio(M, gamma, beta):
    # Calculates the stagnation pressure ratio across the given shock

    x = M*np.sin(beta)
    x2 = x*x
    a = (0.5*(gamma+1.)*x)**2
    b = 1.+0.5*(gamma-1.)*x2
    c = (gamma+1.)*(gamma*x2-0.5*(gamma-1.))**(1./(gamma-1.))
    return 2./c*(a/b)**(gamma/(gamma-1.))


def nu(M, *args):
    # Calculates the Prandtl-Meyer angle for the given flow

    gamma = args[0]

    a = (gamma+1.)/(gamma-1.)
    return np.sqrt(a)*np.arctan(np.sqrt(a**-1*(M**2-1.)))-np.arctan(np.sqrt(M**2-1.))


def dnu_dM(M, *args):
    # Calculates the derivative of the Prandtl-Meyer angle with respect to the Mach number

    gamma = args[0]

    return np.sqrt(M**2-1.)/(M*(1.+0.5*(gamma-1.)*M**2))


def expansion_fan_aft_M(M, gamma, theta):
    # Calculates the Mach number aft of the given expansion fan

    nu_1 = nu(M, gamma)

    # Define root function
    def f(M, *args):
        return nu(M, *args)-theta-nu_1

    # Define error function
    def fe(M, *args):
        return abs(f(M, *args))/M

    # Get solution from Newton's method
    M0 = M*1.1
    M_aft = newtons_method(f, M0, args=(gamma,), dfdx=dnu_dM, fe=fe)

    return M_aft


def C_p_crit(M_inf, gamma):
    # Calculates the critical pressure coefficient as a function of freestream Mach number and gamma

    a = 1.0+0.5*(gamma-1.0)*M_inf**2
    b = (2.0*a/(gamma+1.0))**(gamma/(gamma-1.0))
    return 2/(gamma*M_inf**2)*(b-1.0)


def prandtl_glauert_corr(coef, M_inf):
    # Corrects the given coefficient using the Prandtl-Glauert compressibility correction factor
    return coef*(1.0-M_inf**2)**-0.5


def karman_tsien_corr(coef, M_inf):
    # Corrects the given coefficient using the Karman-Tsien rule

    B = np.sqrt(1.0-M_inf**2)
    return coef/(B+M_inf**2*coef/(2.0*(1.0+B)))


def laitone_corr(coef, M_inf, gamma):
    # Corrects the given coefficient using Laitone's rule

    B = np.sqrt(1.0-M_inf**2)
    R = isentropic_temp_ratio(M_inf, gamma)
    return coef/(B+M_inf**2*coef*R/(2.0*(1.0+B)))


def M_crit(gamma, C_p_min_inc, rule='P-G'):
    # Calculates the critical Mach number as a function of gamma and the minimum incompressible pressure coefficient
    # Uses the false position and secant methods

    # Define root function
    if rule=='P-G':
        def f(M):
            return C_p_crit(M, gamma)-prandtl_glauert_corr(C_p_min_inc, M)
    elif rule=='K-T':
        def f(M):
            return C_p_crit(M, gamma)-karman_tsien_corr(C_p_min_inc, M)
    elif rule=='L':
        def f(M):
            return C_p_crit(M, gamma)-laitone_corr(C_p_min_inc, M, gamma)

    # Try secant method
    M0 = 0.6
    M1 = 0.7
    f0 = f(M0)
    f1 = f(M1)
    max_iter = 1000
    iteration = 0

    while abs(f1) > 1e-12 and iteration < max_iter:

        # Get new guess
        M2 = M1-f1*(M0-M1)/(f0-f1)
        
        # Reseed for bad values
        if M2 >= 1.0:
            M2 = 1.0-abs(M1-M0)*np.random.random()
        if M2 <= 0.0:
            M2 = 0.01+abs(M1-M0)*np.random.random()

        # update
        M0 = M1
        M1 = M2
        f0 = f1
        f1 = f(M2)
        iteration += 1

    return M2


def RK4(f, x0, t0, t1, dt):
    """Integrates f(t,x) from t0 to t1 in steps of dt.
    
    Parameters
    ----------
    f : callable
        Function to integrate.

    x0 : ndarray
        Initial state for dependent variables.
    
    t0 : float
        Initial state for independent variable.
    
    t1 : float
        Final state for independent variable.

    dt : float
        Step size for independent variable.
    """

    # Initialize storage
    t = np.arange(t0, t1, dt)
    N = len(t)
    x = np.zeros((N,len(x0)))
    x[0,:] = x0

    k1 = 0.0
    k2 = 0.0
    k3 = 0.0
    k4 = 0.0
    ti = 0.0
    xi = np.zeros(len(x0))

    a = 0.166666666666666666666666667*dt

    # Loop
    for i in range(N-1):

        ti = t[i]
        xi = x[i,:]

        k1 = f(ti, xi)
        k2 = f(ti+0.5*dt, xi+0.5*dt*k1)
        k3 = f(ti+0.5*dt, xi+0.5*dt*k2)
        k4 = f(ti+dt, xi+dt*k3)

        x[i+1,:] = xi + a*(k1 + 2.0*k2 + 2.0*k3 + k4)

    return t, x