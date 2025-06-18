#import sys, os
#sys.path.insert(0, os.path.dirname(__file__))

import pytest
import numpy as np
from undi import MuonNuclearInteraction

angtom=1.0e-10 # m

# This is a linear F-mu-F along z
r=1.17 * angtom
fmuf = [
            {'Position': np.array([0.000000  ,  0.  ,  0]),
            'Label': 'F',
            },

            {'Position': np.array([0.000000  ,  0.  ,  r ]),
            'Label': 'mu'
            },

            {'Position': np.array([0.000000  ,  0.  ,  2*r]),
            'Label': 'F',
            }
        ]


def brewer(t, r, gamma_F):
    from numpy import cos, sin, sqrt
    h=6.6260693e-34 # Js
    hbar=h/(2*np.pi) # Js
    mu_0=(4e-7)*np.pi # Tm A-1

    gamma_mu = 2*np.pi*135.5e6
    #gamma_F  = 2*np.pi*40.053e6

    omegad = (mu_0*gamma_mu*gamma_F*(hbar))
    omegad /=(4*np.pi*((r)**3))

    tomegad=t*omegad
    y = (1./6.)*(3+cos(sqrt(3)*tomegad)+ \
                (1-1/sqrt(3))*cos(((3-sqrt(3))/2)*tomegad)+ \
                (1+1/sqrt(3))*cos(((3+sqrt(3))/2)*tomegad))
    return y


def test_fmuf():

    steps = 100
    t = np.linspace(0, 16e-6, steps)


    NS = MuonNuclearInteraction(fmuf, log_level='info')
    NS.translate_rotate_sample_vec([0,0,1])
    signal_FmuF = NS.polarization(t, cutoff=1.1 * r)

    NS = MuonNuclearInteraction(fmuf, log_level='info')
    NS.translate_rotate_sample_vec([0,1,0])
    signal_FmuF += NS.polarization(t, cutoff=1.1 * r)

    NS = MuonNuclearInteraction(fmuf, log_level='info')
    NS.translate_rotate_sample_vec([1,0,0])
    signal_FmuF += NS.polarization(t, cutoff=1.1 * r)

    gF = NS.atoms[0]['Gamma']

    np.testing.assert_allclose(signal_FmuF/3, brewer(t, r, gF), atol=2e-6)


