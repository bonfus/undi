#import sys, os
#sys.path.insert(0, os.path.dirname(__file__))

import pytest
import numpy as np
from undi import MuonNuclearInteraction

angtom=1.0e-10 # m
a=4.0 # some lattice constant, in Angstrom

small_atoms = [
    
    {'Position': np.array([0., 0., 0.]),
     'Label': 'mu'},

    {'Position': np.array([0.0    ,    0.5   ,     0.5])*angtom*a,
     'Label': '93Nb'
    },
    
    {'Position': np.array([0.5    ,    0.0   ,     0.5])*angtom*a,
     'Label': '6Li',
    },
    
    {'Position': np.array([0.5    ,    0.5   ,     0.0])*angtom*a,
     'Label': '123Sb',
    },
    
    {'Position': np.array([0.5    ,    0.5   ,     0.5])*angtom*a,
    'Label': '149Sm',
    },
    
]



def test_small_system():
    

    steps = 100
    t = np.linspace(0, 16e-6, steps)
    
    NS = MuonNuclearInteraction(small_atoms, log_level='info')

    np.random.seed(12)
    dpthin = NS.celio_on_steroids(t,  k=1, single_precision=False, algorithm='light')
    #np.savetxt('/tmp/dpthin', dpthin)

    np.random.seed(12)
    spthin = NS.celio_on_steroids(t,  k=1, single_precision=True, algorithm='light')

    np.random.seed(12)
    dpfat = NS.celio_on_steroids(t,  k=1, single_precision=False, algorithm='fat')
    #np.savetxt('/tmp/dpthin', dpthin)

    np.random.seed(12)
    spfat = NS.celio_on_steroids(t,  k=1, single_precision=True, algorithm='fat')
    #np.savetxt('/tmp/spthin', spthin)
    
    np.testing.assert_allclose(spfat, dpfat, atol=1e-6)

    np.testing.assert_allclose(spfat, spthin, atol=1e-6)
    np.testing.assert_allclose(dpfat, dpthin, atol=1e-6)
    
    #np.savetxt('/tmp/aaaa', signal_Cu)


