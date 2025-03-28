# -*- coding: utf-8 -*-
# Licensed under a 3-clause BSD style license - see LICENSE.rst
import logging
import os
import warnings
from collections import OrderedDict
from multiprocessing import Pool

import numpy as np
from astropy import units as u
from astropy.constants import alpha, c, e, hbar, m_e, m_p, sigma_sb
from astropy.utils.data import get_pkg_data_filename

from .extern.validator import (
    validate_array,
    validate_physical_type,
    validate_scalar,
)
from .model_utils import memoize
from .utils import trapz_loglog

__all__ = [
    "Synchrotron",
    "InverseCompton"
]

mec2 = (m_e * c**2).cgs
mec2_unit = u.Unit(mec2)

ar = (4 * sigma_sb / c).to("erg/(cm3 K4)")  # costante di radiazione
r0 = (e**2 / mec2).to("cm")  #raggio classico dell'elettrone 

def _validate_ene(ene):
    from astropy.table import Table

    if isinstance(ene, dict) or isinstance(ene, Table):
        try:
            ene = validate_array(
                "energy", u.Quantity(ene["energy"]), physical_type="energy"
            )
        except KeyError:
            raise TypeError("Table or dict does not have 'energy' column")
    else:
        if not isinstance(ene, u.Quantity):
            ene = u.Quantity(ene)
        validate_physical_type("energy", ene, physical_type="energy")

    return ene

