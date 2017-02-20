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
    fv = [0]*t
    fv[0] = np.matrix([[0.5], [0.5]])
    for i in range(1, t):
        print("\nDay", i,)
        fv[i] = forward(fv[i-1], ev[i-1])
        print(fv[i])


def get_O(umbrella):
    if umbrella:
        return O_true
    else:
        return O_false


forward_algorithm(5)
