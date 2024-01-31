"""Implements a method to calculate the kernel density estimation of the number
of particles with nonzero p_t for a jet with optional saving to a file.
"""

import logging
import pickle
from epicgan.utils import calc_multiplicities


logger = logging.getLogger("main")
