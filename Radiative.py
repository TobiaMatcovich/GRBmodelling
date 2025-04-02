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

from .Validator import (
    validate_array,
    validate_physical_type,
    validate_scalar,
)
#from .model_utils import memoize
from .Utils import trapz_loglog

__all__ = [
    "Synchrotron",
    "InverseCompton"
]

# Get a new logger to avoid changing the level of the astropy logger
log = logging.getLogger("naima.radiative")
log.setLevel(logging.INFO)

e = e.gauss
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

    #  @memoize
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

        spectrum = self._spectrum(photon_energy)

        if distance != 0:
            distance = validate_scalar("distance", distance, physical_type="length")
            flux =spectrum/ 4 * np.pi * distance.to("cm") ** 2
            out_unit = "1/(s cm2 eV)"
        else:
            out_unit = "1/(s eV)"

        return flux.to(out_unit)

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
    """Implements gamma and nelec properties"""

    def __init__(self, particle_distribution):
        super().__init__(particle_distribution)
        self.param_names = ["Eemin", "Eemax", "nEed"]

    @property
    def _gamma(self):
        """Lorentz factor array"""
        log10gmin = np.log10(self.Eemin / mec2).value
        log10gmax = np.log10(self.Eemax / mec2).value
        return np.logspace(
            log10gmin, log10gmax, max(10, int(self.nEed * (log10gmax - log10gmin)))
        )

    @property
    def _nelec(self):
        """Particles per unit lorentz factor"""
        pd = self.particle_distribution(self._gamma* mec2)
        return pd.to(1 / mec2_unit).value
    
    @property
    def Etot(self):
        """Total energy in electrons used for the radiative calculation"""
        Etot = trapz_loglog(self._gamma * self._nelec, self._gamma * mec2)
        return Etot

    def compute_Etot(self, Eemin=None, Eemax=None):
        """Total energy in electrons between energies Eemin and Eemax

        Parameters
        ----------
        Eemin : :class:`~astropy.units.Quantity` float, optional
            Minimum electron energy for energy content calculation.

        Eemax : :class:`~astropy.units.Quantity` float, optional
            Maximum electron energy for energy content calculation.
        """
        if Eemin is None and Eemax is None:
            Etot = self.Etot
        else:
            if Eemax is None:
                Eemax = self.Eemax
            if Eemin is None:
                Eemin = self.Eemin

            log10gmin = np.log10(Eemin / mec2).value
            log10gmax = np.log10(Eemax / mec2).value
            gamma = np.logspace(
                log10gmin, log10gmax, max(10, int(self.nEed * (log10gmax - log10gmin)))
            )
            nelec = self.particle_distribution(gamma * mec2).to(1 / mec2_unit).value
            Etot = trapz_loglog(gamma * nelec, gamma * mec2)

        return Etot
    
    
    def set_Etot(self,Etot, Eemin=None, Eemax=None, amplitude_name=None):
        
        """Normalize particle distribution so that the total energy in electrons
        between Eemin and Eemax is Etot

        Parameters
        ----------
        Etot : :class:`~astropy.units.Quantity` float
            Desired energy in electrons.

        Eemin : :class:`~astropy.units.Quantity` float, optional
            Minimum electron energy for energy content calculation.

        Eemax : :class:`~astropy.units.Quantity` float, optional
            Maximum electron energy for energy content calculation.

        amplitude_name : str, optional
            Name of the amplitude parameter of the particle distribution. It
            must be accesible as an attribute of the distribution function.
            Defaults to ``amplitude``.
        """

        Etot = validate_scalar("Etot", Etot, physical_type="energy")
        oldEtot = self.compute_Etot(Eemin=Eemin, Eemax=Eemax)

        if amplitude_name is None:
            try:
                self.particle_distribution.amplitude *= (Etot / oldEtot).decompose()
            except AttributeError:
                log.error(
                    "The particle distribution does not have an attribute"
                    " called amplitude to modify its normalization: you can"
                    " set the name with the amplitude_name parameter of set_Etot"
                )
        else:
            oldampl = getattr(self.particle_distribution, amplitude_name)
            setattr(
                self.particle_distribution,
                amplitude_name,
                oldampl * (Etot / oldEtot).decompose(),  # decompose in fondamental units
            )
    
    
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
    
    
class InverseCompton(BaseElectron):
    """Inverse Compton emission from an electron population.

    If you use this class in your research, please consult and cite
    `Khangulyan, D., Aharonian, F.A., & Kelner, S.R.  2014, Astrophysical
    Journal, 783, 100 <http://adsabs.harvard.edu/abs/2014ApJ...783..100K>`_

    Parameters
    ----------
    particle_distribution : function
        Particle distribution function, taking electron energies as a
        `~astropy.units.Quantity` array or float, and returning the particle
        energy density in units of number of electrons per unit energy as a
        `~astropy.units.Quantity` array or float.

    seed_photon_fields : string or iterable of strings (optional)
        A list of gray-body or non-thermal seed photon fields to use for IC
        calculation. Each of the items of the iterable can be either:

        * A string equal to radiation fields:
        ``CMB`` (default, Cosmic Microwave Background),2.72 K, energy densitiy of 0.261 eV/cm³
        ``NIR`` (Near Infrared Radiation),  30 K, energy densitiy 0.5 eV/cm³
        ``FIR`` (Far Infrared Radiation), 3000 K,energy densitiy 1 eV/cm³
        (these are the GALPROP values for a location at a distance of 6.5 kpc from the galactic center).

        * A list of length three (isotropic source) or four (anisotropic
          source) composed of:

            1. A name for the seed photon field.
            2. Its temperature (thermal source) or energy (monochromatic or
               non-thermal source) as a :class:`~astropy.units.Quantity`
               instance.
            3. Its photon field energy density as a
               :class:`~astropy.units.Quantity` instance.
            4. Optional: The angle between the seed photon direction and the
               scattered photon direction as a :class:`~astropy.units.Quantity`
               float instance.

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
        distribution arrays. Default is 300.
    """

    def __init__(self, particle_distribution, seed_photon_fields=["CMB"], **kwargs):
        super().__init__(particle_distribution)
        self.seed_photon_fields = self._process_input_seed(seed_photon_fields)
        self.Eemin = 1 * u.GeV
        self.Eemax = 1e9 * mec2
        self.nEed = 100
        self.param_names += ["seed_photon_fields"]
        self.__dict__.update(**kwargs)
        
        
    @staticmethod
    def _process_input_seed(seed_photon_fields):
        """
        take input list of seed_photon_fields and fix them into usable format
        """

        Tcmb = 2.72548 * u.K  # 0.00057 K
        Tfir = 30 * u.K
        ufir = 0.5 * u.eV / u.cm**3
        Tnir = 3000 * u.K
        unir = 1.0 * u.eV / u.cm**3

        # Allow for seed_photon_fields definitions of the type 'CMB-NIR-FIR' or
        # 'CMB')
            
        if not isinstance(seed_photon_fields, list):
            seed_photon_fields = seed_photon_fields.split("-") 


        result = OrderedDict()

        for idx, inseed in enumerate(seed_photon_fields):
            seed = {}
            if isinstance(inseed, str):
                name = inseed
                seed["type"] = "thermal"
                if inseed == "CMB":
                    seed["T"] = Tcmb
                    seed["u"] = ar * Tcmb**4
                    seed["isotropic"] = True
                elif inseed == "FIR":
                    seed["T"] = Tfir
                    seed["u"] = ufir
                    seed["isotropic"] = True
                elif inseed == "NIR":
                    seed["T"] = Tnir
                    seed["u"] = unir
                    seed["isotropic"] = True
                else:
                    log.warning(
                        "Will not use seed {0} because it is not "
                        "CMB, FIR or NIR".format(inseed)
                    )
                    raise TypeError
                
            elif type(inseed) is list and (len(inseed) == 3 or len(inseed) == 4):
                
                # if len==3 is isotropic instead is not beacuse it has also the angle 
                isotropic = len(inseed) == 3

                if isotropic:
                    name, T, uu = inseed
                    seed["isotropic"] = True
                else:
                    name, T, uu, theta = inseed
                    seed["isotropic"] = False
                    seed["theta"] = validate_scalar(
                        "{0}-theta".format(name), theta, physical_type="angle"
                    )

                thermal = T.unit.physical_type == "temperature"

                if thermal:
                    seed["type"] = "thermal"
                    validate_scalar(
                        "{0}-T".format(name),
                        T,
                        domain="positive",
                        physical_type="temperature",
                    )
                    seed["T"] = T
                    if uu == 0:
                        seed["u"] = ar * T**4
                    else:
                        # pressure has same physical type as energy density
                        validate_scalar(
                            "{0}-u".format(name),
                            uu,
                            domain="positive",
                            physical_type="pressure",
                        )
                        seed["u"] = uu
                else:
                    seed["type"] = "array"
                    # Ensure everything is in arrays
                    T = u.Quantity((T,)).flatten()
                    uu = u.Quantity((uu,)).flatten()

                    seed["energy"] = validate_array(
                        "{0}-energy".format(name),
                        T,
                        domain="positive",
                        physical_type="energy",
                    )

                    if np.isscalar(seed["energy"]) or seed["energy"].size == 1:
                        seed["photon_density"] = validate_scalar(
                            "{0}-density".format(name),
                            uu,
                            domain="positive",
                            physical_type="pressure",
                        )
                    else:
                        if uu.unit.physical_type == "pressure":
                            uu /= seed["energy"] ** 2
                        seed["photon_density"] = validate_array(
                            "{0}-density".format(name),
                            uu,
                            domain="positive",
                            physical_type="differential number density",
                        )
            else:
                raise TypeError(
                    "Unable to process seed photon field: {0}".format(inseed)
                )

            result[name] = seed

        return result