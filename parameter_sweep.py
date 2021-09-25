"""Search parameter space for chaotic solutions in 2-player RPS with
self-play. Output a .csv file with Lyapunov exponents for each parameter set.

Check a grid of points, as well as random points in the tested subspace. This
is important to confirm that chaotic solutions are not due to some numerical
peculiarity"""

import symengine
import numpy as np
from datetime import datetime
from itertools import product
from scipy.stats import sem
from jitcode import jitcode_lyap, y

SCATTER = True
SLICES = 5 ## Number of grid points per dimension in parameter space
START, STOP = 1, 22 # limits of testing region in parameter space

time_string = datetime.now().strftime("%Y-%m-%d-%H:%M")
filename = 'parameters_test_' + time_string + '.csv'
if SCATTER:
    filename = "Scatter_" + filename

columns = ["parameters", "lyap_1", "lyap_2", "lyap_3", "lyap_4",
           "lyap_5", "lyap_6", "stderr_1", "stderr_2", "stderr_3",
           "stderr_4", "stderr_5", "stderr_6"]

columns_string = ','.join(columns) + "\n"

with open(filename, "w+") as f:
    f.write(columns_string)

#FILE = './chaos_parameters_test.csv'

PAYOFFS = (1.0, np.sqrt(2), 15.4, 1.0)

parameters_a = np.linspace(START, STOP, SLICES)
if SCATTER:
    parameters_b = START + (STOP - START) * np.random.random(SLICES)
    parameters_c = START + (STOP - START) * np.random.random(SLICES)
else:
    parameters_b = np.linspace(START, STOP, SLICES)
    parameters_c = np.linspace(START, STOP, SLICES)

payoffs_list = product(parameters_a, parameters_b, parameters_c)

def gen_axes_parameters():
    a_slices = np.linspace(1.0,5.5,4)
    c_coordinates = np.linspace(-2.0,2.0,10)
    b_coordinates = np.linspace(-2.0,2.0,10)
    cs = product(np.linspace(1, 22, 15), c_coordinates)
    bs = product(b_coordinates, np.linspace(1, 22, 15))
    axes = list(bs) + list(cs)
    params = [(x,y,z) for x in a_slices for (y,z) in axes]
    return params

payoffs_list = gen_axes_parameters()

def main():
    for payoffs in payoffs_list:
        a = 1.0
        b, c, d = payoffs
        payoffs_line = ' '.join(map(str,[a,b,c,d]))
        mean_0 = symengine.Symbol("mean_0")
        mean_1 = symengine.Symbol("mean_1")
        helpers = [(mean_0, c * (y(0) * (y(5) - y(4)) +
                                 y(1) * (y(3) - y(5)) +
                                 y(2) * (y(4) - y(3)))),
                   (mean_1, d * (y(3) * (y(2) - y(1)) +
                                 y(4) * (y(0) - y(2)) +
                                 y(5) * (y(1) - y(0))))]
        
        f = [
                y(0) * ((a * y(2)) + (-a * y(1))) 
                + y(0) * ((c * y(5)) + (-c * y(4)) - mean_0),

                y(1) * ((a * y(0)) + (-a * y(2))) 
                + y(1) * ((c * y(3)) + (-c * y(5)) - mean_0),

                y(2) * ((a * y(1)) 
                + (-a * y(0))) + y(2) * ((c * y(4)) + (-c * y(3)) - mean_0),

                y(3) * ((b * y(5)) 
                + (-b * y(4))) + y(3) * ((d * y(2)) + (-d * y(1)) - mean_1),

                y(4) * ((b * y(3)) + (-b * y(5))) 
                + y(4) * ((d * y(0)) + (-d * y(2)) - mean_1), 

                y(5) * ((b * y(4)) + (-b * y(3))) 
                + y(5) * ((d * y(1)) + (-d * y(0)) - mean_1)
            ]
        
        initial_state = np.array([0.5, 0.2, 0.3, 0.2, 0.5, 0.3])
        
        n = len(f)
        ODE = jitcode_lyap(f, helpers=helpers, n_lyap=n)
        ODE.set_integrator("dopri5")
        ODE.set_initial_value(initial_state, 0.0)
        
        times = np.linspace(0, 300, 20000)
        lyaps = []
        data = []
        for time in times:
            lyaps.append(ODE.integrate(time)[1])
        
        lyaps = np.vstack(lyaps)
        saved_lyaps = []
        stderrs = []
        for i in range(n):
            lyap = np.average(lyaps[1000:,i])
            stderr = sem(lyaps[1000:,i]) # estimate
            saved_lyaps.append(lyap)
            stderrs.append(stderr)
            print("%i. Lyapunov exponent: % .4f ± %.4f" %(i+1, lyap, stderr))
        lyaps_str = ','.join(map(str, saved_lyaps))
        stderrs_str = ','.join(map(str,stderrs))
        with open(filename, "a") as f:
            line = payoffs_line + ',' + lyaps_str + ',' + stderrs_str + '\n'
            f.write(line)

#if __name__ == "__main__":
#    main()
