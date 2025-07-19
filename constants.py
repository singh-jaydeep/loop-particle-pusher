import numpy as np
from numba import njit


## basic sim parameters
eps = 0.01  ## sets the charge to mass ratio (size 1/eps), and the initial loop radius (eps)
total_t = 9000
dt = eps / 5
total_iter = int(total_t / dt)  ## verify this is at least num_record
num_record = 100
recordinterval_iter = int(total_iter / num_record)
num_part = 150  ## total number of particles
theta_sep = (
    2 * 3.14159 / num_part
)  ## initial angular separation between adjacent particles
RECORDING = True  ## False only really useful for debugging

## run simulation with time-steps in parallel for different particles.
PARALLEL = False


## where data is stored
data_path = "data/simul_data.json"

## starting loop center
init_x = 2
init_y = 2


## planar magnetic field
@njit
def bfield_mag(x, y):
    return 10 * x


## Current options are "RK4" and "BORIS". If using RK4, ensure dt is sufficiently small in terms of eps for stability
alg = "BORIS"
