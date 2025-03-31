# -*- coding: utf-8 -*-
# Licensed under a 3-clause BSD style license - see LICENSE.rst
import os

import astropy.units as u
import numpy as np
from astropy.table import Table
from astropy.utils.data import get_pkg_data_filename

from .extern.validator import (
    validate_array,
    validate_physical_type,
    validate_scalar,
)
from .model_utils import memoize
from .radiative import Bremsstrahlung, InverseCompton, PionDecay, Synchrotron

__all__ = [
    "BrokenPowerLaw",
    "ExponentialCutoffPowerLaw",
    "PowerLaw",
    "LogParabola",
    "ExponentialCutoffBrokenPowerLaw",
]

def _validate_ene(ene):
    if isinstance(ene, dict) or isinstance(ene, Table):
        try:
            ene = validate_array(
                "energy", u.Quantity(ene["energy"]), physical_type="energy"
            )
        except KeyError:
            raise TypeError("Table or dict does not have 'ene' column")
    else:
        if not isinstance(ene, u.Quantity):
            ene = u.Quantity(ene)
        validate_physical_type("energy", ene, physical_type="energy")

    return ene

class PowerLaw:
    """
    One dimensional power law model.

    Parameters
    ----------
    amplitude : float
        Model amplitude.
    e_0 : `~astropy.units.Quantity` float
        Reference energy
    alpha : float
        Power law index

    See Also
    --------
    PowerLaw, BrokenPowerLaw, LogParabola

    Notes
    -----
    Model formula: f(E) = A (E / E_0) ^ {-\\alpha}

    """
    
    param_names = ["amplitude", "e_0", "alpha"]
    _memoize = False
    
    def __init__(self, amplitude, e_0, alpha):
        self.amplitude = amplitude
        self.e_0 = validate_scalar("e_0", e_0, domain="positive", physical_type="energy")
        self.alpha = alpha
        
    @staticmethod
    def eval(e, amplitude, e_0, alpha):
        """One dimensional power law model function"""

        xx = e / e_0
        return amplitude * xx ** (-alpha)
    
    
    # @memoize non definito qui ma in NAIMA si
    def _calc(self, e):
        return self.eval(
            e.to("eV").value,
            self.amplitude,
            self.e_0.to("eV").value,
            self.alpha,
        )
        
    def __call__(self, e):
        e = _validate_ene(e)
        return self._calc(e)
    
    
class ExponentialCutoffPowerLaw:
    """
    One dimensional power law model with an exponential cutoff.

    Parameters
    ----------
    amplitude : float
        Model amplitude
    e_0 : `~astropy.units.Quantity` float
        Reference point
    alpha : float
        Power law index
    e_cutoff : `~astropy.units.Quantity` float
        Cutoff point
    beta : float
        Cutoff exponent

    Notes
    -----
    Model formula: f(E) = A (E / E_0) ^ {-alpha}*exp (- (E / E_{cutoff}) ^ beta)

    """

    param_names = ["amplitude", "e_0", "alpha", "e_cutoff", "beta"]


    def __init__(self, amplitude, e_0, alpha, e_cutoff, beta=1.0):
        self.amplitude = amplitude
        self.e_0 = validate_scalar(
            "e_0", e_0, domain="positive", physical_type="energy"
        )
        self.alpha = alpha
        self.e_cutoff = validate_scalar(
            "e_cutoff", e_cutoff, domain="positive", physical_type="energy"
        )
        self.beta = beta

    @staticmethod
    def eval(e, amplitude, e_0, alpha, e_cutoff, beta):
        "One dimensional power law with an exponential cutoff model function"

        xx = e / e_0
        return amplitude * xx ** (-alpha) * np.exp(-((e / e_cutoff) ** beta))

    #  @memoize
    def _calc(self, e):
        return self.eval(
            e.to("eV").value,
            self.amplitude,
            self.e_0.to("eV").value,
            self.alpha,
            self.e_cutoff.to("eV").value,
            self.beta,
        )

    def __call__(self, e):
        e = _validate_ene(e)
        return self._calc(e)
    
    
class BrokenPowerLaw:
    """
    One dimensional power law model with a break.

    Parameters
    ----------
    amplitude : float
        Model amplitude at the break energy
    e_0 : `~astropy.units.Quantity` float
        Reference point
    e_break : `~astropy.units.Quantity` float
        Break energy
    alpha_1 : float
        Power law index for x < x_break
    alpha_2 : float
        Power law index for x > x_break


    Notes
    -----
    Model formula (two cases):
    
    f(E) = \\left \\{
                     \\begin{array}{ll}
                       A (E / E_0) ^ {-\\alpha_1} & : E < E_{break} \\\\
                       A (E_{break}/E_0) ^ {\\alpha_2-\\alpha_1}
                           (E / E_0) ^ {-\\alpha_2} & :  E > E_{break} \\\\
                     \\end{array}
                   \\right.
    """

    param_names = ["amplitude", "e_0", "e_break", "alpha_1", "alpha_2"]

    def __init__(self, amplitude, e_0, e_break, alpha_1, alpha_2):
        self.amplitude = amplitude
        self.e_0 = validate_scalar(
            "e_0", e_0, domain="positive", physical_type="energy"
        )
        self.e_break = validate_scalar(
            "e_break", e_break, domain="positive", physical_type="energy"
        )
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2

    @staticmethod
    def eval(e, amplitude, e_0, e_break, alpha_1, alpha_2):
        
        K = np.where(e < e_break, 1, (e_break / e_0) ** (alpha_2 - alpha_1))
        alpha = np.where(e < e_break, alpha_1, alpha_2)
        return amplitude * K * (e / e_0) ** -alpha

    # @memoize
    def _calc(self, e):
        
        return self.eval(
            e.to("eV").value,
            self.amplitude,
            self.e_0.to("eV").value,
            self.e_break.to("eV").value,
            self.alpha_1,
            self.alpha_2,
        )

    def __call__(self, e):
        
        e = _validate_ene(e)
        return self._calc(e)
    
class ExponentialCutoffBrokenPowerLaw:
    """
    One dimensional power law model with a break.

    Parameters
    ----------
    amplitude : float
        Model amplitude at the break point
    e_0 : `~astropy.units.Quantity` float
        Reference point
    e_break : `~astropy.units.Quantity` float
        Break energy
    alpha_1 : float
        Power law index for x < x_break
    alpha_2 : float
        Power law index for x > x_break
    e_cutoff : `~astropy.units.Quantity` float
        Exponential Cutoff energy
    beta : float, optional
        Exponential cutoff rapidity. Default is 1.

    See Also
    --------
    PowerLaw, ExponentialCutoffPowerLaw, LogParabola

    Notes
    -----
    Model formula (two case):

            f(E) = \\exp(-(E / E_{cutoff})^\\beta)\\left \\{
                     \\begin{array}{ll}
                       A (E / E_0) ^ {-\\alpha_1}    & : E < E_{break} \\\\
                       A (E_{break}/E_0) ^ {\\alpha_2-\\alpha_1}
                            (E / E_0) ^ {-\\alpha_2} & : E > E_{break} \\\\
                     \\end{array}
                   \\right.

    """

    param_names = [
        "amplitude",
        "e_0",
        "e_break",
        "alpha_1",
        "alpha_2",
        "e_cutoff",
        "beta",
    ]

    def __init__(self, amplitude, e_0, e_break, alpha_1, alpha_2, e_cutoff, beta=1.0):
        self.amplitude = amplitude
        self.e_0 = validate_scalar(
            "e_0", e_0, domain="positive", physical_type="energy"
        )
        self.e_break = validate_scalar(
            "e_break", e_break, domain="positive", physical_type="energy"
        )
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.e_cutoff = validate_scalar(
            "e_cutoff", e_cutoff, domain="positive", physical_type="energy"
        )
        self.beta = beta

    @staticmethod
    def eval(e, amplitude, e_0, e_break, alpha_1, alpha_2, e_cutoff, beta):
        
        K = np.where(e < e_break, 1, (e_break / e_0) ** (alpha_2 - alpha_1))
        alpha = np.where(e < e_break, alpha_1, alpha_2)
        ee2 = e / e_cutoff
        return amplitude * K * (e / e_0) ** -alpha * np.exp(-(ee2**beta))

    #  @memoize
    def _calc(self, e):
        return self.eval(
            e.to("eV").value,
            self.amplitude,
            self.e_0.to("eV").value,
            self.e_break.to("eV").value,
            self.alpha_1,
            self.alpha_2,
            self.e_cutoff.to("eV").value,
            self.beta,
        )

    def __call__(self, e):

        e = _validate_ene(e)
        return self._calc(e)