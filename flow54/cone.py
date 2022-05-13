import numpy as np
from compressible_tools import *


class Cone:
    """Class for calculating axisymmetric conical flows in supersonic flow.
    
    Parameters
    ----------
    theta : float
        Cone half angle.
    """

    def __init__(self, theta):

        self.theta = theta


    def solve_taylor_maccoll(self, M):
        """Solves the Taylor-Maccoll equation over the cone.
        
        Parameters
        ----------
        M : float
            Freestream Mach number.
        """

        pass