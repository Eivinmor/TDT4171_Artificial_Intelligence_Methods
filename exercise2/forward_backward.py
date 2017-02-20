import numpy as np

T = np.matrix([[0.7, 0.3],      # Transition model
               [0.3, 0.7]])

O_true = np.matrix([[0.9, 0],   # Observation model
                    [0, 0.2]])

O_false = np.matrix([[0.1, 0],  # Observation model
                     [0, 0.8]])

ev = [1, 1, 0, 1, 1]            # Evidence


def forward(prev_msg, evidence):
    prediction = get_O(evidence) * np.transpose(T) * prev_msg   # O_t * T(transposed) * f_{t-1}
    normalisation = prediction.sum()                            # Calculate normalisation
    return prediction / normalisation                           # Return normalised values


def forward_algorithm(t):
    t += 1
    fv = [None]*t
    fv[0] = np.matrix([[0.5], [0.5]])
    for i in range(1, t):
        fv[i] = forward(fv[i-1], ev[i-1])
    return fv


def backward(prev_msg, evidence):
    return T * get_O(evidence) * prev_msg                  # T * O_t * b_{k+2:t}


def forward_backward_algorithm(t):
    fv = forward_algorithm(t)
    sv = [None]*t
    b = np.matrix([[1], [1]])
    for i in range(t, 0, -1):
        print("\nDay", i)
        sv[i-1] = np.multiply(fv[i], b)     # Calculate sv
        sv[i-1] /= sv[i-1].sum()            # Normalise sv
        b = backward(b, ev[i-1])              # Save backward message
        print("sv:\n", sv[i-1])
        print("b:\n", b)


def get_O(umbrella):       # Returns the correct observation matrix given the current evidence
    if umbrella:
        return O_true
    else:
        return O_false


forward_backward_algorithm(5)
