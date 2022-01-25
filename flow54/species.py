import numpy as np

class Species:
    """Container class for a gas species. Defaults to properties of air.

    Parameters
    ----------
    M_w : float
        Molecular weight.

    gamma : float
        Ratio of specific heats.
    """

    def __init__(self, M_w=28.96443, gamma=1.4):

        # Store params
        self.M_w = M_w
        self.gamma = gamma

        # Universal gas constant
        self.R_u = 8314.4612 # J/kmol K

        # Calculate derived properties
        self.R_g = self.R_u/self.M_w # J/kg K
        self.c_v = 1.0/(gamma-1.0)*self.R_g
        self.c_p = gamma*self.c_v


class MixedSpecies(Species):
    """Container class for a mixed gas species.

    Parameters
    ----------
    ratios : ndarray
        Volume/molar fractions for the components.

    M_w : ndarray
        Molecular weights for the components.

    c_p : ndarray
        Mass-specific heats for the components.
    """

    def __init__(self, ratios, M_w, c_p):

        # Universal gas constant
        self.R_u = 8314.4612 # J/kmol K

        # Average molecular weight
        self.M_w = np.sum(ratios*M_w).item()

        # Specific gas constant
        self.R_g = self.R_u/self.M_w

        # Average specific heat
        self.c_p = np.sum(ratios*M_w*c_p).item()/self.M_w

        # Derived properties
        self.c_v = self.c_p-self.R_g
        self.gamma = self.c_p/self.c_v