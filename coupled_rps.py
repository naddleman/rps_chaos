"""
Demo functions to help explain what's going on with the dynamics. The JITCODE
version is more concise and faster, but the verbose version might be more clear.

This file also includes various useful functions for visualization:

poincare_scatter() produces a Poincare map for a particular parameter set

poincare_sequence() produces an animation to visualize how solutions change
as a parameter is adjusted

"""

import numpy as np 
import matplotlib.pyplot as plt
from scipy.integrate import ode, odeint
from matplotlib import animation

sample_payoffs = [1, 1, 1.5, 1]

def RPS_rep_dyn(X, t=0, a=1, growth=False):
    """ +/- a is the payoffs for winning/losing """
    if growth:
        b = a * 1.5
    else: b = a
    mean_fitness = (X[0] * X[1] + X[0] * X[2] + X[1] * X[2]) * (a - b)
    
    arr = np.array([ X[0] * (-b * X[1] + a * X[2] - mean_fitness),
                     X[1] * (-b * X[2] + a * X[0] - mean_fitness),
                     X[2] * (-b * X[0] + a * X[1] - mean_fitness)])

    return arr

def RPS_two_players(X, t=0, payoffs=[1,3 * np.sqrt(2)]):
    """
    A, B are payoff matrices for each population
    """
    mat = np.array([[0, -1, 1],
                    [1, 0, -1],
                    [-1, 1, 0]])

    pop1, pop2 = X[:3], X[3:]
    A, B = (mat * x for x in payoffs)
    d1 = pop1 * ((A @ pop2) - pop1 @ A @ pop2)
    d2 = pop2 * ((B @ pop1) - pop2 @ B @ pop1)
    return np.concatenate((d1, d2))

def RPS_two_players_self_play(X, t=0, payoffs=sample_payoffs,
                              invert=False):
    """
    the elements of payoffs are pop1, pop2 rewards from self-play, and then 
    pop1 & pop2's rewards from cross-population play.

    invert = True if you want to reverse the dominance in cross-population play
    """
    mat = np.array([[0, -1, 1],
                    [1, 0, -1],
                    [-1, 1, 0]])

    pop1, pop2 = X[:3], X[3:]
    payoff_matrices = [mat * x for x in payoffs]
    if invert:
        payoff_matrices[2] *= -1
        payoff_matrices[3] *= -1
    d1 = pop1 * ((payoff_matrices[2] @ pop2) - pop1 @ payoff_matrices[2] @ pop2)
    d2 = pop2 * ((payoff_matrices[3] @ pop1) - pop2 @ payoff_matrices[3] @ pop1)
    i1 = RPS_rep_dyn(pop1, a=payoffs[0])
    i2 = RPS_rep_dyn(pop2, a=payoffs[1])
    return np.concatenate((d1 + i1, d2 + i2))

def RPS_rep_dyn_coupled(X, t, N=2, payoffs=[1,2], coupling=0):
    """ payoffs is a positive vector of winning payoffs in each system
        it must have length N
        0 < coupling <= N is the strength of interaction between systems
        it is normalized by number of interacting groups
    """
    assert len(payoffs) == N
    interaction = coupling / N
    # first determine the within-population effect
    within_arr = np.zeros(3 * N)
    for i in range(N):
        payoff = payoffs[i]
        X_within = X[3*i:3*(i+1)]
        arr = RPS_rep_dyn(X_within, t=t, a = payoff)
        within_arr[3*i:3*(i+1)] = arr
    # Now add between-pops effect

    interaction_array = np.zeros(3 * N)

    for i in range(3):
        # Make a vector [0, 1, 0, 0, 1, 0], e.g., to adjust population values
        strategy_index = [0] * i + [1] + [0] * (2 - i)
        strategy_indices = np.array(strategy_index * N)
        strategy_populations = X[i::3]
        mean_for_strategy = np.mean(strategy_populations)
        differences_from_mean = mean_for_strategy - strategy_populations 
        interaction_array[i::3] = differences_from_mean * interaction

    #return within_arr, interaction_array
    return np.sum((within_arr, interaction_array), axis=0)

def RPS_rep_dyn_external(X, t, N=2, payoffs=[1,2], coupling=0.5, reverse=False):
    """
    Play RPS internally and externally. "reverse" the dominance to make R>P, etc

    across-pop payoff is +/-1 
    """
    assert len(payoffs) == N
    interaction = coupling / N
    for i in range(N):
        payoff = payoffs[i]
        X_within = X[3*i:3*(i+1)]
        arr = RPS_rep_dyn(X_within, t=t, a = payoff)
        within_arr[3*i:3*(i+1)] = arr
    # Now add between-pops effect
    interaction_array = np.zeros(3 * N)

    for i in range(n):
        for j in range(i):
            pass

        # Make a vector [0, 1, 0, 0, 1, 0], e.g., to adjust population values
        strategy_index = [0] * i + [1] + [0] * (2 - i)
        strategy_indices = np.array(strategy_index * N)
        strategy_populations = X[i::3]
        mean_for_strategy = np.mean(strategy_populations)
        differences_from_mean = mean_for_strategy - strategy_populations 
        interaction_array[i::3] = differences_from_mean * interaction

    
e_freqs = np.array([1, 2, 3, 4])

def plot_2_pop(coupling_factor):
    N = 2
    payoffs = np.array([1., np.sqrt(3)])
    coupling = coupling_factor
    init = np.array([0.5, 0.25, 0.25] * 2)
    t = np.linspace(0,150,20000)
    rps_c,output = odeint(RPS_rep_dyn_coupled,
                   init,
                   t,
                   args=(N, payoffs, coupling), full_output = 1)
           
    plt.plot(t, np.transpose([rps_c[:,0], rps_c[:,3]]))
    distance_from_c = rps_c[:,0:3] - (1/3)
    plt.show()
    return rps_c
    
    #plt.plot(t, np.sum(distance_from_c, axis=1))
    #plt.show()

parameters = {"N":4,
              "payoffs": np.array([1,np.sqrt(2), np.sqrt(3),np.sqrt(5)]),
              "coupling": 0.5,
              "init": [0.5, 0.25, 0.25]}

def plot_by_parameters(parameters):
    N = parameters["N"]
    payoffs = parameters["payoffs"]
    coupling = parameters["coupling"]
    init = np.array(parameters["init"] * N)
    t = np.linspace(0,300,1200000)
    rps_c = odeint(RPS_rep_dyn_coupled,
                   init,
                   t,
                   args=(N, payoffs, coupling))
    plt.plot(t, np.transpose([rps_c[:,3*i] for i in range(N)]))
    plt.show()
    plt.plot(t, rps_c[:,0])
    plt.show()

def demo_bimatrix():
    t = np.linspace(0,200,100000)
    pop = np.array([0.5, 0.2, 0.3, 0.2, 0.5, 0.3])
    rps_c = odeint(RPS_two_players, pop, t)
    plt.plot(t, rps_c[:,0], label="Rock1")
    plt.plot(t, rps_c[:,3], label="Rock2")
    plt.legend()
    plt.show()

def demo_self_play(payoffs = [1.4, 1.6, .5, .5]):
    t = np.linspace(0,10000,200000)
    invert = False
    pop = np.array([0.5, 0.2, 0.3, 0.2, 0.5, 0.3])
    rps_cs = odeint(RPS_two_players_self_play,
                    pop,
                    t,
                    args=(payoffs, invert))
    plt.plot(t, rps_cs[:,0], label="Rock1")
    plt.plot(t, rps_cs[:,3], label="Rock2")
    plt.title("Payoffs" + str(payoffs))

    plt.legend()
    plt.show()
    return rps_cs

def self_play_chaos_test(d, payoffs = [1, np.sqrt(2), 15.4, 1]):
    t = np.linspace(0,300, 20000)
    invert = False
    pop1 = np.array([0.5, 0.2, 0.3, 0.2, 0.5, 0.3])
    pop2 = pop1 + [d, -d, 0, 0, 0, 0]
    rps_cs1 = odeint(RPS_two_players_self_play,
                     pop1,
                     t,
                     args=(payoffs, invert))
    rps_cs2 = odeint(RPS_two_players_self_play,
                     pop2,
                     t,
                     args=(payoffs, invert))
    squared_diffs = (rps_cs1 - rps_cs2) ** 2
    distance = np.sqrt(np.sum(squared_diffs, axis=1))
    #plt.plot(t, rps_cs1[:,0], label = "Pop 1 rock")
    plt.plot(t, distance, label = "distance")
    #diffs = rps_cs1[:,0] - rps_cs2[:,0]
    #plt.plot(t, diffs, label = "R1 - perturbed")
    plt.title(F"Euclidean distance between orbits\ninitial separation = {d}")
    plt.ylabel("distance")
    plt.xlabel("time")
    plt.legend()
    plt.show()


def self_play(payoffs):
    t = np.linspace(0,1000,20000)
    invert = False
    pop = np.array([0.5, 0.2, 0.3, 0.2, 0.5, 0.3])
    rps_cs = odeint(RPS_two_players_self_play,
                    pop,
                    t,
                    args=(payoffs, invert))
    return rps_cs

def poincare_scatter(orbit):
    """
    Finds points x0, y1 where -0.01 < x1 - x0 + y1 - y0 < 0.01
    orbit has length 6 for 2-population scenario.
    """
    diffs = orbit[:,1] - orbit[:,0] + orbit[:,4] - orbit[:,3]
    indices = np.abs(diffs) < 0.01
    outs = orbit[indices]
    plt.plot(orbit[:,0], orbit[:,4], ',')
    plt.ylabel("y1")
    plt.xlabel("x0")
    plt.title("Poincare section abs(x1 - x0 + y1 - y0) < .01")
    plt.show()

# demo_self_play([1, 1.1, 0.4, 1]) is good
    
def poincare_sequence(payoffs, filename):
    """
    Shows how poincare map varies as a parameter is varied

    For now just varies the third parameter by adding [0, 1]

    """
    fig, ax = plt.subplots()
    initial_payoffs = np.array(payoffs)

    points, = ax.plot([], [], ',')
    def init():
        points.set_data([], [])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        return points,

    def animate(i):
        current_payoff = initial_payoffs + [0, 0, i*0.025, 0]
        
        orbit = self_play(current_payoff)
        points.set_data(orbit[:,0], orbit[:,4])
        ax.set_title("payoffs: " + str(current_payoff))
        return points,
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=61, blit=False)

    anim.save('./' + filename, writer='animation.FFMpegWriter', fps=3)


#t = np.linspace(0, 700, 10000)
#t_rps = np.linspace (0, 20, 10000)
#X0 = np.array([0, 0])
#
#N = 2
#payoffs = np.array([1,2])
#coupling = 0
#rps_init = np.array([0.25, 0.25, 0.5])
#rps_coupled_init = np.array([0.5, 0.25, 0.25] * 2)
#rps_c, rps_c_info = odeint(RPS_rep_dyn_coupled, rps_coupled_init,
#                           t_rps, args=(N, payoffs, coupling), full_output=True)
##rps, rps_info = odeint(RPS_rep_dyn, rps_init, t_rps, full_output=True)
#plt.plot(t_rps, np.transpose([rps_c[:,0], rps_c[:,3]]))
#plt.show()
