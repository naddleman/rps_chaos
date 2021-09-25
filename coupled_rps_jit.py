""" Implement dynamics with jitcode

"""
import numpy as np
import symengine
from jitcode import jitcode, y

PAYOFFS = (1.0, np.sqrt(2), 15.4, 1.0)
a, b, c, d = PAYOFFS
mean_0 = symengine.Symbol("mean_0")
mean_1 = symengine.Symbol("mean_1")
helpers = [(mean_0, c * (y(0) * (y(5) - y(4)) +
                         y(1) * (y(3) - y(5)) +
                         y(2) * (y(4) - y(3)))),
           (mean_1, d * (y(3) * (y(2) - y(1)) +
                         y(4) * (y(0) - y(2)) +
                         y(5) * (y(1) - y(0))))]

f = [
        y(0) * ((a * y(2)) + (-a * y(1))) + y(0) * ((c * y(5)) + (-c * y(4)) - mean_0),
        y(1) * ((a * y(0)) + (-a * y(2))) + y(1) * ((c * y(3)) + (-c * y(5)) - mean_0),
        y(2) * ((a * y(1)) + (-a * y(0))) + y(2) * ((c * y(4)) + (-c * y(3)) - mean_0),
        y(3) * ((b * y(5)) + (-b * y(4))) + y(3) * ((d * y(2)) + (-d * y(1)) - mean_1),
        y(4) * ((b * y(3)) + (-b * y(5))) + y(4) * ((d * y(0)) + (-d * y(2)) - mean_1), 
        y(5) * ((b * y(4)) + (-b * y(3))) + y(5) * ((d * y(1)) + (-d * y(0)) - mean_1)

    ]
initial_state = np.array([0.5, 0.2, 0.3, 0.2, 0.5, 0.3])
ODE = jitcode(f, helpers=helpers)
ODE.set_integrator("dopri5")
ODE.set_initial_value(initial_state, 0.0)

times = np.linspace(0, 300, 20000)
data = []
for time in times:
    data.append(ODE.integrate(time))

