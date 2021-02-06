import numpy as np


def hr_egn(A, B, R, x0):
    """Takes the adjacency matrix, payment matrix (or matrices), relationship matrix and initial state
       of all vertices to return the increments in each strategy for each vertex"""
    # A  - Adjacency matrix, np.ndarray (N,N)
    # B  - A 2D or 3D matrix with all payoff matrices, np.ndarray (S,S,N)
    # R  - Relationship or preference matrix, np.ndarray (N,N)
    # x0 - Initial state of our system, np.ndarray (N,S), must be double

    # Number of players
    N = A[:, 0].size
    # Number of strategies
    S = x0[0, :].size
    # Degree and degree of preferences
    d = np.zeros([N, 2])
    d[:, 0] = np.dot(A, np.ones(N))

    for v in range(N):
        d[v, 1] = np.dot(np.ceil(np.abs(R[v, :])), A[v, :])

    # Player v neighborhood
    k = np.zeros([N, S], dtype='double')
    for v in range(N):
        for u in range(N):
            k[v, :] = np.add(k[v, :], np.multiply(A[v, u], x0[u, :]))
        # Weights the neighborhood
        k[v, :] = np.multiply(np.divide(1, d[v, 0]), k[v, :])

    # This variable is the increments that x0 receives, the derivative
    x = np.zeros([N, S], dtype='double')
    # This is the unit vector with 1 in some entry
    es = np.zeros(S, dtype='int')

    # Phi and gamma
    p = 0
    g = 0

    # Auxiliary variables for better clarity
    aux1 = 0
    aux2 = 0

    # Here is the derivative calculation
    # We first test if all payoffs are the same so we do less comparisons
    if B.ndim == 2:
        for v in range(N):
            for s in range(S):
                # Set es value
                es[s] = 1
                for u in range(N):
                    if v == u:
                        # Same payoff personal equation
                        # First we will do the dot products
                        # e_s*B*k_v
                        aux1 = np.dot(es, np.dot(B, k[v, :]))
                        # x_v*B*k_v
                        aux2 = np.dot(x0[v, :], np.dot(B, k[v, :]))
                        # Finally we subtract them to multiply by r_vv
                        p = np.multiply(R[v, u], np.subtract(aux1, aux2))
                    elif A[v, u] != 0:
                        # Same payoff social equation
                        # x_u*B*e_s
                        aux1 = np.dot(x0[u, :], np.dot(B, es))
                        # x_u*B*x_v
                        aux2 = np.dot(x0[u, :], np.dot(B, x0[v, :]))
                        # Subtract then multiply
                        aux1 = np.subtract(aux1, aux2)
                        aux2 = np.multiply(R[v, u], A[v, u])
                        g = np.add(g, np.multiply(aux2, aux1))
                # Weights the social part
                if d[v, 1] != 0:
                    g = np.multiply(np.divide(1, d[v, 1]), g)
                # Estimates the derivative
                x[v, s] = np.multiply(x0[v, s], np.add(p, g))
                # Prepare variables to next iteration
                p = 0
                g = 0
                es[s] = 0
    else:
        for v in range(N):
            for s in range(S):
                # Same thing as before, but now with individual payoffs
                es[s] = 1
                for u in range(N):
                    if v == u:
                        # Individual payoffs personal equation
                        # e_s*B_v*k_v
                        aux1 = np.dot(es, np.dot(B[:, :, v], k[v, :]))
                        # x_u*B_v*k_v
                        aux2 = np.dot(x0[v, :], np.dot(B[:, :, v], k[v, :]))
                        p = np.multiply(R[v, u], np.subtract(aux1, aux2))
                    elif A[v, u] != 0:
                        # Individual payoffs social equation
                        # x_u*B_u*e_s
                        aux1 = np.dot(x0[u, :], np.dot(B[:, :, u], es))
                        # x_u*B_u*x_v
                        aux2 = np.dot(x0[u, :], np.dot(B[:, :, u], x0[v, :]))
                        # Subtract then multiply
                        aux1 = np.subtract(aux1, aux2)
                        aux2 = np.multiply(R[v, u], A[v, u])
                        g = np.add(g, np.multiply(aux2, aux1))
                # Weights the social part
                if d[v, 1] != 0:
                    g = np.multiply(np.divide(1, d[v, 1]), g)
                # Estimates the derivative
                x[v, s] = np.multiply(x0[v, s], np.add(p, g))
                # Prepare variables to next iteration
                p = 0
                g = 0
                es[s] = 0
    return x


def hr_game(t0, tf, n, A, B, R, x0):
    """Takes a time interval, number of steps, adjacency matrix, payment matrix (or matrices),
       relationship matrix and initial state of all vertices to return the result of the
       hyper-rational evolutionary game played on the given graph in that time interval"""
    # t0 - Initial time
    # tf - Final time
    # n - Number of steps
    # A - Adjacency matrix, np.ndarray (N,N)
    # B  - A 2D or 3D matrix with all payoff matrices, np.ndarray (S,S,N)
    # R  - Relationship or preference matrix, np.ndarray (N,N)
    # x0 - Initial state of our system, np.ndarray (N,S), must be double

    # Number of players
    N = A[:, 0].size
    # Number of strategies
    S = B[:, 0].size
    # Step in each iteration
    h = (tf - t0) / n
    # Result of each step, np.ndarray (N, S, n+1)
    y = np.zeros([N, S, n+1], dtype='double')
    y[:, :, 0] = x0
    k = np.zeros([N, S])

    # Fourth order Runge-Kutta
    for t in range(n):
        k1 = np.multiply(h, hr_egn(A, B, R, y[:, :, t]))
        k2 = np.multiply(h, hr_egn(A, B, R, np.add(y[:, :, t], np.divide(k1, 2))))
        k3 = np.multiply(h, hr_egn(A, B, R, np.add(y[:, :, t], np.divide(k2, 2))))
        k4 = np.multiply(h, hr_egn(A, B, R, np.add(y[:, :, t], k3)))
        # k = (k1 + 2*k2 + 2*k3 + k4)/6
        k = np.divide(np.add(np.add(k1, np.multiply(2, k2)), np.add(np.multiply(2, k3), k4)), 6)

        y[:, :, t+1] = np.add(y[:, :, t], k)

        # Filter results with machine epsilon
        for v in range(N):
            for s in range(S):
                if y[v, s, t+1] < np.sqrt(np.finfo('double').eps):
                    y[v, s, t+1] = 0
                elif y[v, s, t+1] > np.subtract(1, np.sqrt(np.finfo('double').eps)):
                    y[v, s, t + 1] = 1

    return y
