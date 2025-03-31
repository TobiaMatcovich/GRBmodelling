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
#from .model_utils import memoize
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


class BaseRadiative:
    """Base class for radiative models

    This class implements the flux, sed methods and subclasses must implement
    the spectrum method which returns the intrinsic differential spectrum.
    """

    def __init__(self, particle_distribution):
        self.particle_distribution = particle_distribution
        try:
            # Check first for the amplitude attribute, which will be present if
            # the particle distribution is a function from naima.models
            pd = self.particle_distribution.amplitude
            validate_physical_type(
                "Particle distribution",
                pd,
                physical_type="differential energy",
            )
        except (AttributeError, TypeError):
            # otherwise check the output
            pd = self.particle_distribution([0.1, 1, 10] * u.TeV)
            validate_physical_type(
                "Particle distribution",
                pd,
                physical_type="differential energy",
            )

    @memoize
    def flux(self, photon_energy, distance=1 * u.kpc):
        """Differential flux at a given distance from the source.

        Parameters
        ----------
        photon_energy : :class:`~astropy.units.Quantity` float or array
            Photon energy array.

        distance : :class:`~astropy.units.Quantity` float, optional
            Distance to the source. If set to 0, the intrinsic differential
            luminosity will be returned. Default is 1 kpc.
        """

        spec = self._spectrum(photon_energy)

        if distance != 0:
            distance = validate_scalar("distance", distance, physical_type="length")
            spec /= 4 * np.pi * distance.to("cm") ** 2
            out_unit = "1/(s cm2 eV)"
        else:
            out_unit = "1/(s eV)"

        return spec.to(out_unit)

    def sed(self, photon_energy, distance=1 * u.kpc):
        """Spectral energy distribution at a given distance from the source.

        Parameters
        ----------
        photon_energy : :class:`~astropy.units.Quantity` float or array
            Photon energy array.

        distance : :class:`~astropy.units.Quantity` float, optional
            Distance to the source. If set to 0, the intrinsic luminosity will
            be returned. Default is 1 kpc.
        """
        if distance != 0:
            out_unit = "erg/(cm2 s)"
        else:
            out_unit = "erg/s"

        photon_energy = _validate_ene(photon_energy)

        sed = (self.flux(photon_energy, distance) * photon_energy**2.0).to(out_unit)

        return sed
    
    
    
class BaseElectron(BaseRadiative):
    """Implements gam and nelec properties"""

    def __init__(self, particle_distribution):
        super().__init__(particle_distribution)
        self.param_names = ["Eemin", "Eemax", "nEed"]
        self._memoize = True
        self._cache = {}  # non chiaro 
        self._queue = []  # non chiaro
        
    @property
    def _gam(self):
        """Lorentz factor array"""
        log10gmin = np.log10(self.Eemin / mec2).value
        log10gmax = np.log10(self.Eemax / mec2).value
        return np.logspace(
            log10gmin, log10gmax, max(10, int(self.nEed * (log10gmax - log10gmin)))
        )

    @property
    def _nelec(self):
        """Particles per unit lorentz factor"""
        pd = self.particle_distribution(self._gam * mec2)
        return pd.to(1 / mec2_unit).value
    
    @property
    def We(self):
        """Total energy in electrons used for the radiative calculation"""
        We = trapz_loglog(self._gam * self._nelec, self._gam * mec2)
        return We

    def compute_We(self, Eemin=None, Eemax=None):
        """Total energy in electrons between energies Eemin and Eemax

        Parameters
        ----------
        Eemin : :class:`~astropy.units.Quantity` float, optional
            Minimum electron energy for energy content calculation.

        Eemax : :class:`~astropy.units.Quantity` float, optional
            Maximum electron energy for energy content calculation.
        """
        if Eemin is None and Eemax is None:
            We = self.We
        else:
            if Eemax is None:
                Eemax = self.Eemax
            if Eemin is None:
                Eemin = self.Eemin

            log10gmin = np.log10(Eemin / mec2).value
            log10gmax = np.log10(Eemax / mec2).value
            gam = np.logspace(
                log10gmin, log10gmax, max(10, int(self.nEed * (log10gmax - log10gmin)))
            )
            nelec = self.particle_distribution(gam * mec2).to(1 / mec2_unit).value
            We = trapz_loglog(gam * nelec, gam * mec2)

        return We
    
    
    
    
class Synchrotron(BaseElectron):
    """Synchrotron emission from an electron population.

    This class uses the approximation of the synchrotron emissivity in a
    random magnetic field of Aharonian, Kelner, and Prosekin 2010, PhysRev D
    82, 3002 (`arXiv:1006.1045 <http://arxiv.org/abs/1006.1045>`_).

    Parameters
    ----------
    particle_distribution : function
        Particle distribution function, taking electron energies as a
        `~astropy.units.Quantity` array or float, and returning the particle
        energy density in units of number of electrons per unit energy as a
        `~astropy.units.Quantity` array or float.

    B : :class:`~astropy.units.Quantity` float instance, optional
        Isotropic magnetic field strength. Default: equipartition
        with CMB (3.24e-6 G)

    Other parameters
    ----------------
    Eemin : :class:`~astropy.units.Quantity` float instance, optional
        Minimum electron energy for the electron distribution. Default is 1
        GeV.

    Eemax : :class:`~astropy.units.Quantity` float instance, optional
        Maximum electron energy for the electron distribution. Default is 510
        TeV.

    nEed : scalar
        Number of points per decade in energy for the electron energy and
        distribution arrays. Default is 100.
    """
    
    def __init__(self, particle_distribution, B=3.24e-6 * u.G, **kwargs):
        super().__init__(particle_distribution)
        self.B = validate_scalar("B", B, physical_type="magnetic flux density")
        self.Eemin = 1 * u.GeV
        self.Eemax = 1e9 * mec2
        self.nEed = 100
        self.param_names += ["B"]
        self.__dict__.update(**kwargs)
        
    def _spectrum(self, photon_energy):
        """Compute intrinsic synchrotron differential spectrum for energies in
        ``photon_energy``

        Compute synchrotron for random magnetic field according to
        approximation of Aharonian, Kelner, and Prosekin 2010, PhysRev D 82,
        3002 (`arXiv:1006.1045 <http://arxiv.org/abs/1006.1045>`_)."""
        
        validated_energy = _validate_ene(photon_energy)
        from scipy.special import cbrt
        
        def Gtilde(x):
            """
            AKP10 Eq. D7

            Factor ~2 performance gain in using cbrt(x)**n vs x**(n/3.)
            Invoking crbt only once reduced time by ~40%
            """
            cb = cbrt(x)
            gt1 = 1.808 * cb / np.sqrt(1 + 3.4 * cb**2.0)
            gt2 = 1 + 2.210 * cb**2.0 + 0.347 * cb**4.0
            gt3 = 1 + 1.353 * cb**2.0 + 0.217 * cb**4.0
            return gt1 * (gt2 / gt3) * np.exp(-x)
        
        Num= np.sqrt(3) * e.value**3 * self.B.to("G").value
        Den = (2 * np.pi * m_e.cgs.value* c.cgs.value**2* hbar.cgs.value* validated_energy.to("erg").value)
        factor=Num/Den
        
        # Critical energy in erg 
        Ec = (3 * e.value * hbar.cgs.value * self.B.to("G").value * self._gam**2)/ (2 * (m_e * c).cgs.value)
        
        EgEc=validated_energy.to("erg").value/np.vstack(Ec)        
        dNdE = factor * Gtilde(EgEc)
        spectrum = (trapz_loglog(np.vstack(self._nelec) * dNdE, self._gam, axis=0) / u.s / u.erg )
        spectrum = spectrum.to("1/(s eV)")
        
        return spectrum