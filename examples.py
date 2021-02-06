import numpy as np
import matplotlib.pyplot as plt
import hrgames as hrg
import sys


def classic_pd():
    """Plays a classic prisoner's dilemma between two players"""
    # Two connected players
    A = np.zeros([2, 2])
    A[0, 1] = 1
    A[1, 0] = 1
    # Classical positive prisoner's dilemma
    B = np.zeros([2, 2])
    B[0, 0] = 3  # R
    B[0, 1] = 1  # S
    B[1, 0] = 4  # T
    B[1, 1] = 2  # P
    # Relationship matrix
    R = np.zeros([2, 2], dtype='double')
    R[0, 0] = 2/3  # + 0.05
    R[0, 1] = 1/3  # - 0.05
    R[1, 1] = 2/3  # + 0.05
    R[1, 0] = 1/3  # - 0.05
    # Initial Condition, 0.5 in all strategies for all players
    xini = np.divide(np.ones([2, 2], dtype='double'), 2)
    # Time interval and number of steps
    t0 = 0
    tf = 50
    n = (tf-t0)*10
    h = (tf-t0)/n

    x = hrg.hr_game(t0, tf, n, A, B, R, xini)

    # Plot results
    xaxis = np.arange(t0, tf+h, h)
    p1 = x[0, 0, :]
    p2 = x[1, 0, :]

    plt.plot(xaxis, p1, 'r', label='Player 1')
    plt.plot(xaxis, p2, 'b', label='Player 2')
    plt.legend(['Player 1', 'Player 2'])
    plt.title("Cooperation probability")
    plt.show()
    plt.close()
    print(x[0, :, n], x[1, :, n])
    print(np.allclose(np.add(x[0, 0, n], x[0, 1, n]), 1), np.allclose(np.add(x[1, 0, n], x[1, 1, n]), 1))

    return None


def classic_pd_negative():
    """Plays a classic prisoner's dilemma between two players"""
    # Two connected players
    A = np.zeros([2, 2])
    A[0, 1] = 1
    A[1, 0] = 1
    # Classical positive prisoner's dilemma
    B = np.zeros([2, 2])
    B[0, 0] = -2  # R
    B[0, 1] = -7  # S
    B[1, 0] = 0  # T
    B[1, 1] = -5  # P
    # Relationship matrix
    R = np.zeros([2, 2], dtype='double')
    R[0, 0] = 5/7 - 0.05
    R[0, 1] = 2/7 + 0.05
    R[1, 1] = 5/7 + 0.05
    R[1, 0] = 2/7 - 0.05
    # Initial Condition, 0.5 in all strategies for all players
    xini = np.divide(np.ones([2, 2], dtype='double'), 2)
    # Time interval and number of steps
    t0 = 0
    tf = 150
    n = (tf-t0)*10
    h = (tf-t0)/n

    x = hrg.hr_game(t0, tf, n, A, B, R, xini)

    # Plot results
    xaxis = np.arange(t0, tf+h, h)
    p1 = x[0, 0, :]
    p2 = x[1, 0, :]

    plt.plot(xaxis, p1, 'r', label='Player 1')
    plt.plot(xaxis, p2, 'b', label='Player 2')
    plt.legend(['Player 1', 'Player 2'])
    plt.title("Cooperation probability")
    plt.show()
    plt.close()
    print(x[0, :, n], x[1, :, n])

    return None


if __name__ == '__main__':
    # Run the example receive as parameter
    locals()[sys.argv[1]]()
