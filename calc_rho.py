# coding: utf8

import time

from msspec.calculator import MSSPEC
from msspec.utils import hemispherical_cluster, get_atom_index
from ase.build  import bulk
from ase.visualize import view



a0 = 3.925 # The lattice parameter in angstroms
kinetic_energy = 500       # Kenitic energy

# Create the copper cubic cell

copper = bulk('Pt', a=a0, cubic=True)

cluster = hemispherical_cluster(copper, planes=3, emitter_plane=2)


# Set the absorber (the deepest atom centered in the xy-plane)

cluster.absorber = get_atom_index(cluster, 0, 0, 0)


# Create a calculator for the PhotoElectron Diffration
#calc = MSSPEC(spectroscopy='EIG', algorithm='inversion')



# Create a calculator for the spectral radius
calc = MSSPEC(spectroscopy='PED', algorithm='inversion')
#calc = MSSPEC(spectroscopy='PED', algorithm='expansion')

# Here is how to tweak the scattering order
#calc.calculation_parameters.scattering_order = 2


#calc.calculation_parameters.renormalization_mode = "Pi_1"
#calc.calculation_parameters.renormalization_omega = 0.3950280116944187 + 0.09735609125775824j 

# Set the cluster to use for the calculation

calc.set_atoms(cluster)


# Run the calculation

#data = calc.get_eigen_values(kinetic_energy=317)

data = calc.get_theta_scan(level='2p',kinetic_energy=kinetic_energy)

#data = calc.get_eigen_values(kinetic_energy=kinetic_energy)

#data = calc.get_eigen_values(kinetic_energy=all_ke)
# Show the results

#data.view()

# Save the results to files
#data.save('results.hdf5')
data.export('exported_results')

# Clean temporary files

calc.shutdown()


