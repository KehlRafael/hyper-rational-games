import numpy as np
import matplotlib.pyplot as plt
import hrgames as hrg
import sys


def classic_pd(R=None, tf=None, xini=None):
    """Plays a classic prisoner's dilemma between two players"""
    # Two connected players
    A = np.zeros([2, 2])
    A[0, 1] = 1
    A[1, 0] = 1
    # Classic positive prisoner's dilemma
    B = np.zeros([2, 2])
    B[0, 0] = 3  # R
    B[0, 1] = 1  # S
    B[1, 0] = 4  # T
    B[1, 1] = 2  # P
    # Relationship matrix
    if R is None or R.shape != (2, 2):
        R = np.zeros([2, 2], dtype='double')
        R[0, 0] = 2/3 + 0.05
        R[0, 1] = 1/3 - 0.05
        R[1, 1] = 2/3 - 0.05
        R[1, 0] = 1/3 + 0.05
    # Initial Condition, 0.5 in all strategies for all players
    if xini is None or xini.shape != (6, 2):
        xini = np.divide(np.ones([2, 2], dtype='double'), 2)
    # Time interval and number of steps
    t0 = 0
    if tf is None:
        tf = 150
    n = (tf-t0)*10
    h = (tf-t0)/n

    x = hrg.hr_game(t0, tf, n, A, B, R, xini)

    # Plot results
    xaxis = np.arange(t0, tf+h, h)

    plt.plot(xaxis, x[0, 0, :], 'r', label='Player 1')
    plt.plot(xaxis, x[1, 0, :], 'b', label='Player 2', alpha=0.7)
    plt.ylim([-0.05, 1.05])
    plt.legend(['Player 1', 'Player 2'])
    plt.title("Cooperation probability")
    plt.show()
    plt.close()

    return None


def classic_pd_negative(R=None, tf=None, xini=None):
    """Plays a classic prisoner's dilemma between two players"""
    # Two connected players
    A = np.zeros([2, 2])
    A[0, 1] = 1
    A[1, 0] = 1
    # Classic negative prisoner's dilemma
    B = np.zeros([2, 2])
    B[0, 0] = -2  # R
    B[0, 1] = -7  # S
    B[1, 0] = 0  # T
    B[1, 1] = -5  # P
    # Relationship matrix
    if R is None or R.shape != (2, 2):
        R = np.zeros([2, 2], dtype='double')
        R[0, 0] = 5/7 - 0.05
        R[0, 1] = 2/7 + 0.05
        R[1, 1] = 5/7 + 0.05
        R[1, 0] = 2/7 - 0.05
    # Initial Condition, 0.5 in all strategies for all players
    if xini is None or xini.shape != (2, 2):
        xini = np.divide(np.ones([2, 2], dtype='double'), 2)
    # Time interval and number of steps
    t0 = 0
    if tf is None:
        tf = 150
    n = (tf-t0)*10
    h = (tf-t0)/n

    x = hrg.hr_game(t0, tf, n, A, B, R, xini)

    # Plot results
    xaxis = np.arange(t0, tf+h, h)

    plt.plot(xaxis, x[0, 0, :], 'r', label='Player 1')
    plt.plot(xaxis, x[1, 0, :], 'b', label='Player 2', alpha=0.7)
    plt.ylim([-0.05, 1.05])
    plt.legend(['Player 1', 'Player 2'])
    plt.title("Cooperation probability")
    plt.show()
    plt.close()

    return None


def hinge_love_triangle(R=None, tf=None, xini=None):
    """Plays a love triangle game in a 'hinge' graph, a path with 3 nodes"""
    # A hinge, path with 3 nodes
    A = np.zeros([3, 3])
    A[0, 1] = 1
    A[1, 0] = 1
    A[1, 2] = 1
    A[2, 1] = 1
    # Payoff matrices
    # Person 1 wants to play s1 against Person 2
    # Person 2 only benefits when playing s2 against Person 1
    # Person 2 has no benefit from playing with Person 3
    B = np.zeros([2, 2, 3])
    # Player 1 payoff matrix
    B[0, 0, 0] = 0
    B[0, 1, 0] = 1
    B[1, 0, 0] = 0
    B[1, 1, 0] = 0
    # Player 2 payoff matrix
    B[0, 0, 1] = 0
    B[0, 1, 1] = 0
    B[1, 0, 1] = 1
    B[1, 1, 1] = 0
    # Player 3 payoff matrix
    B[0, 0, 2] = 1
    B[0, 1, 2] = 0
    B[1, 0, 2] = 0
    B[1, 1, 2] = 0

    # Relationship matrix
    if R is None or R.shape != (3, 3):
        R = np.zeros([3, 3], dtype='double')
        # Person 1 cares equally about itself and it's partner
        R[0, 0] = 1/2
        R[0, 1] = 1/2
        # Person 2 cares about itself and Person 3
        R[1, 1] = 1/2 + 0.05
        R[1, 2] = 1/2 - 0.05
        # Person cares only about itself
        R[2, 2] = 1

    # Initial Condition, 0.5 in all strategies for all players
    if xini is None or xini.shape != (3, 2):
        xini = np.divide(np.ones([3, 2], dtype='double'), 2)
    # Time interval and number of steps
    t0 = 0
    if tf is None:
        tf = 150
    n = (tf - t0) * 10
    h = (tf - t0) / n

    x = hrg.hr_game(t0, tf, n, A, B, R, xini)

    # Plot results
    xaxis = np.arange(t0, tf + h, h)

    plt.plot(xaxis, x[0, 0, :], 'r', label='Player 1')
    plt.plot(xaxis, x[1, 0, :], 'b', label='Player 2', alpha=0.7)
    plt.plot(xaxis, x[2, 0, :], 'g', label='Player 2', alpha=0.7)
    plt.ylim([-0.05, 1.05])
    plt.legend(['Player 1', 'Player 2', 'Player 3'])
    plt.title("Probability of using strategy 1")
    plt.show()
    plt.close()

    return None


def k3_love_triangle(R=None, tf=None, xini=None):
    """Plays a love triangle game in a k3 graph, a complete graph with 3 nodes"""
    # A k3, a complete graph with 3 nodes
    A = np.subtract(np.ones([3, 3]), np.eye(3))
    # Payoff matrices
    # Person 1 wants to play s1 against Person 2
    # Person 2 only benefits when playing s2 against Person 1
    # Person 2 has no benefit from playing with Person 3
    B = np.zeros([2, 2, 3])
    # Player 1 payoff matrix
    B[0, 0, 0] = 0
    B[0, 1, 0] = 1
    B[1, 0, 0] = 0
    B[1, 1, 0] = 0
    # Player 2 payoff matrix
    B[0, 0, 1] = 0
    B[0, 1, 1] = 0
    B[1, 0, 1] = 1
    B[1, 1, 1] = 0
    # Player 3 payoff matrix
    B[0, 0, 2] = 1
    B[0, 1, 2] = 0
    B[1, 0, 2] = 0
    B[1, 1, 2] = 0

    # Relationship matrix
    if R is None or R.shape != (3, 3):
        R = np.zeros([3, 3], dtype='double')
        # Person 1 cares equally about itself and it's partner, but dislikes Person 3
        R[0, 0] = 2/5
        R[0, 1] = 2/5
        R[0, 2] = -1/5
        # Person 2 cares about itself and Person 3
        R[1, 1] = 1/2 - 0.05
        R[1, 2] = 1/2 + 0.05
        # Person cares only about itself
        R[2, 2] = 1

    # Initial Condition, 0.5 in all strategies for all players
    if xini is None or xini.shape != (3, 2):
        xini = np.divide(np.ones([3, 2], dtype='double'), 2)
    # Time interval and number of steps
    t0 = 0
    if tf is None:
        tf = 150
    n = (tf - t0) * 10
    h = (tf - t0) / n

    x = hrg.hr_game(t0, tf, n, A, B, R, xini)

    # Plot results
    xaxis = np.arange(t0, tf + h, h)

    plt.plot(xaxis, x[0, 0, :], 'r', label='Player 1')
    plt.plot(xaxis, x[1, 0, :], 'b', label='Player 2', alpha=0.7)
    plt.plot(xaxis, x[2, 0, :], 'g', label='Player 2', alpha=0.7)
    plt.ylim([-0.05, 1.05])
    plt.legend(['Player 1', 'Player 2', 'Player 3'])
    plt.title("Probability of using strategy 1")
    plt.show()
    plt.close()

    return None


def closed_star_pd(R=None, tf=None, xini=None):
    """Plays a prisoner's dilemma on a closed star with 6 vertices"""
    # Closed star, it's easier to erase connections
    A = np.subtract(np.ones([6, 6]), np.eye(6))
    A[1, 3] = 0
    A[1, 4] = 0
    A[2, 4] = 0
    A[2, 5] = 0
    A[3, 5] = 0
    A[3, 1] = 0
    A[4, 1] = 0
    A[4, 2] = 0
    A[5, 2] = 0
    A[5, 3] = 0
    # Classic positive prisoner's dilemma
    B = np.zeros([2, 2])
    B[0, 0] = 3  # R
    B[0, 1] = 1  # S
    B[1, 0] = 4  # T
    B[1, 1] = 2  # P
    # Relationship matrix, everyone's selfish
    if R is None or R.shape != (6, 6):
        R = np.eye(6)

    # Initial Condition
    if xini is None or xini.shape != (6, 2):
        xini = np.zeros([6, 2], dtype='double')
        # Natural cooperators
        xini[0, 0] = 0.99
        xini[1, 0] = 0.99
        xini[3, 0] = 0.99
        xini[5, 0] = 0.99
        # Natural defectors
        xini[2, 0] = 0.01
        xini[4, 0] = 0.01
        # Iterate to complete assignments
        for i in range(6):
            xini[i, 1] = np.subtract(1, xini[i, 0])

    # Time interval and number of steps
    t0 = 0
    if tf is None:
        tf = 150
    n = (tf-t0)*10
    h = (tf-t0)/n

    x = hrg.hr_game(t0, tf, n, A, B, R, xini)

    # Plot results
    xaxis = np.arange(t0, tf+h, h)
    labels = ["Player " + str(i) for i in range(1, 7)]

    for i in range(6):
        plt.plot(xaxis, x[i, 0, :], label=labels[i], alpha=0.7)
    plt.ylim([-0.05, 1.05])
    plt.legend(labels)
    plt.title("Cooperation probability")
    plt.show()
    plt.close()

    return None


def closed_star_coex(R=None, tf=None, xini=None):
    """Plays a coexistance game on a closed star with 6 vertices"""
    # Closed star, it's easier to erase connections
    A = np.subtract(np.ones([6, 6]), np.eye(6))
    A[1, 3] = 0
    A[1, 4] = 0
    A[2, 4] = 0
    A[2, 5] = 0
    A[3, 5] = 0
    A[3, 1] = 0
    A[4, 1] = 0
    A[4, 2] = 0
    A[5, 2] = 0
    A[5, 3] = 0
    # Basic coexistance game
    B = np.zeros([2, 2])
    B[0, 0] = 0
    B[0, 1] = 1
    B[1, 0] = 0
    B[1, 1] = 1
    # Relationship matrix, everyone's selfish
    if R is None or R.shape != (6, 6):
        R = np.eye(6)

    # Initial Condition
    if xini is None or xini.shape != (6, 2):
        xini = np.zeros([6, 2], dtype='double')
        # Almost 1
        xini[0, 0] = 0.99
        xini[1, 0] = 0.99
        xini[3, 0] = 0.99
        xini[5, 0] = 0.99
        # Almost 0
        xini[2, 0] = 0.01
        xini[4, 0] = 0.01
        # Iterate to complete assignments
        for i in range(6):
            xini[i, 1] = np.subtract(1, xini[i, 0])

    # Time interval and number of steps
    t0 = 0
    if tf is None:
        tf = 150
    n = (tf-t0)*10
    h = (tf-t0)/n

    x = hrg.hr_game(t0, tf, n, A, B, R, xini)

    # Plot results
    xaxis = np.arange(t0, tf+h, h)
    labels = ["Player " + str(i) for i in range(1, 7)]

    for i in range(6):
        plt.plot(xaxis, x[i, 0, :], label=labels[i], alpha=0.7)
    plt.ylim([-0.05, 1.05])
    plt.legend(labels)
    plt.title("Cooperation probability")
    plt.show()
    plt.close()

    return None


def simulations_madeo_mocenni():
    """
    Runs the same simulations found in Madeo and Mocennis article
    but with hyper-rational agents instead of standard replicators.
    There will be some code rewriting because I want all functions
    to be standalone examples and not reusable functions.
    """

    # Lists for iterations
    ini_cond = ['external', 'central', 'ext_cent']
    graph = ['open_star', 'closed_star', 'asym_weighted']

    # Graph definitions
    open_star = np.zeros([6, 6])
    open_star[0, 1] = 1
    open_star[1, 0] = 1
    open_star[0, 2] = 1
    open_star[2, 0] = 1
    open_star[0, 3] = 1
    open_star[3, 0] = 1
    open_star[0, 4] = 1
    open_star[4, 0] = 1
    open_star[0, 5] = 1
    open_star[5, 0] = 1

    closed_star = np.subtract(np.ones([6, 6]), np.eye(6))
    closed_star[1, 3] = 0
    closed_star[1, 4] = 0
    closed_star[2, 4] = 0
    closed_star[2, 5] = 0
    closed_star[3, 5] = 0
    closed_star[3, 1] = 0
    closed_star[4, 1] = 0
    closed_star[4, 2] = 0
    closed_star[5, 2] = 0
    closed_star[5, 3] = 0

    asym_weighted = np.zeros([6, 6])
    asym_weighted[0, 1] = 1
    asym_weighted[0, 2] = 1
    asym_weighted[0, 4] = 3
    asym_weighted[0, 5] = 1
    asym_weighted[1, 0] = 1
    asym_weighted[1, 5] = 3
    asym_weighted[2, 0] = 1
    asym_weighted[2, 3] = 3
    asym_weighted[3, 2] = 3
    asym_weighted[3, 4] = 1
    asym_weighted[4, 3] = 1
    asym_weighted[4, 0] = 3
    asym_weighted[5, 0] = 1
    asym_weighted[5, 1] = 3

    # Save graph matrices
    for i in range(3):
        np.savetxt(graph[i]+'.txt', locals()[graph[i]])

    # Payoffs
    bistability = np.zeros([2, 2])
    bistability[0, 0] = 1
    bistability[1, 1] = 1

    prisoners_dilemma = np.zeros([2, 2])
    prisoners_dilemma[0, 0] = 1
    prisoners_dilemma[1, 0] = 1.5

    coexistence = np.zeros([2, 2])
    coexistence[0, 1] = 1
    coexistence[1, 0] = 1

    # Initial conditions
    external = np.zeros([6, 2], dtype="double")
    for v in range(6):
        if v == 1:
            external[v, :] = np.array([0.01, 0.99])
        else:
            external[v, :] = np.array([0.99, 0.01])

    central = np.zeros([6, 2], dtype="double")
    for v in range(6):
        if v == 0:
            central[v, :] = np.array([0.01, 0.99])
        else:
            central[v, :] = np.array([0.99, 0.01])

    ext_cent = np.zeros([6, 2], dtype="double")
    for v in range(6):
        if v == 0 or v == 1:
            ext_cent[v, :] = np.array([0.01, 0.99])
        else:
            ext_cent[v, :] = np.array([0.99, 0.01])

    # Relationship matrices
    R_eye = np.eye(6)
    R_open_star = rel_matrix_builder(open_star)
    R_closed_star = rel_matrix_builder(closed_star)
    R_asym_weighted = rel_matrix_builder(asym_weighted)

    # Do simulation and print final state do text file
    # Bistability
    for i in range(3):
        for j in range(3):
            x = hrg.hr_game(0, 50, 500, locals()[graph[j]], bistability,
                            locals()['R_'+graph[j]], locals()[ini_cond[i]])
            np.savetxt('bi-'+ini_cond[i]+'-'+graph[j]+'.txt', x[:, :, 500])

    # Prisoner's Dilemma
    for i in range(3):
        for j in range(3):
            x = hrg.hr_game(0, 100, 1000, locals()[graph[j]], prisoners_dilemma,
                            locals()['R_'+graph[j]], locals()[ini_cond[i]])
            np.savetxt('pd-'+ini_cond[i]+'-'+graph[j]+'.txt', x[:, :, 1000])

    # Coexistance
    for i in range(3):
        for j in range(3):
            x = hrg.hr_game(0, 50, 500, locals()[graph[j]], coexistence,
                            locals()['R_'+graph[j]], locals()[ini_cond[i]])
            np.savetxt('co-'+ini_cond[i]+'-'+graph[j]+'.txt', x[:, :, 500])

    return None


def rel_matrix_builder(adj):
    """
    Creates a relationship matrix with 1/2 for self
    and 1/2 divided between neighbors.
    """
    rel = np.zeros_like(adj)
    size = rel.shape[0]
    rel += np.divide(np.eye(size), 2)

    for i in range(size):
        nz = np.count_nonzero(adj[i, :])
        for j in range(size):
            if adj[i, j] != 0:
                rel[i, j] = np.divide(np.divide(1, 2), nz)

    return rel


if __name__ == '__main__':
    # Run the example receive as parameter
    locals()[sys.argv[1]]()
