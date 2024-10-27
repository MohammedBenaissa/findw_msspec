# coding: utf-8
#
# Source file  : <Will be updated during commit>
# Last modified: <Will be updated during commit>
# Committed by : <Will be updated during commit>

#coding: utf-8

import numpy as np

from ase.build import bulk

from msspec.calculator import MSSPEC
from msspec.iodata import Data
from msspec.utils import hemispherical_cluster, get_atom_index




a0 = 3.925                 # The lattice parameter in angstroms
kinetic_energy = 500       # Kenitic energy


# Create the copper cubic cell
system = bulk('Pt', a=a0, cubic=True)
cluster = hemispherical_cluster(system, planes=3, emitter_plane=2)
# Set the absorber (the deepest atom centered in the xy-plane)
cluster.absorber = get_atom_index(cluster, 0, 0, 0)
# Create a calculator for the PhotoElectron Diffration
#calc = MSSPEC(spectroscopy='PED', algorithm='inversion')


# Set the cluster to use for the calculation
savefile = "rho.hdf5"
# Create a calculator for the spectral radius
calc = MSSPEC(spectroscopy='EIG', algorithm='inversion')
calc.calculation_parameters.renormalization_mode = "Pi_1"
calc.calculation_parameters.renormalization_omega = 0.37271073106250613 + -0.21559938271496626j
# Set the cluster to use for the calculation
calc.set_atoms(cluster)

# Run the calculation
data = calc.get_eigen_values(kinetic_energy=kinetic_energy)
#data = calc.get_eigen_values(kinetic_energy=all_ke)


# Save results to hdf5 file
data.save(savefile)
