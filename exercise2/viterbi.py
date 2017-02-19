import numpy as np

T = np.matrix([[0.7, 0.3],      # Transition model
               [0.3, 0.7]])

O_true = np.matrix([[0.9, 0],   # Observation model for U_t = true
                    [0, 0.2]])

O_false = np.matrix([[0.1, 0],  # Observation model for U_t = false
                     [0, 0.8]])

ev = [None, 0, 0, 1, 1, 0]      # Evidence vector


def forward(prev_msg, evidence):
    prediction = get_O(evidence) * np.transpose(T) * prev_msg   # O_t * T(transposed) * f_{t-1}
    normalisation = prediction.sum()                            # Calculate normalisation
    return prediction / normalisation                           # Return normalised values


def viterbi(t):
    # Initialisation
    mv = [None]*t   # List of matrices with messages in format [[R_t = true],[R_t = false]]
    mv[0] = forward(np.matrix([[0.5], [0.5]]), ev[1])     # Performing initial step (filtering)
    print("\n", mv[0])

    # Algorithm
    for i in range(1, t):
        mv[i] = calc_viterbi_message(mv[i-1], ev[i+1])
        print("\n", mv[i])

    # Printing most probable path
    print("\nMost likely states: ")
    for state_probs in mv:
        print(state_probs[0, 0] > state_probs[1, 0], end=", ")
    print("\b\b")


def calc_viterbi_message(prev_msg, evidence):
    path_probs = np.multiply(prev_msg.transpose(), T)   # Calculate probabilities of the paths
    max_path_probs = path_probs.max(1)                  # Select max path probabilities per current state (matrix row)
    message = get_O(evidence) * max_path_probs          # Multiply with observation probabilities
    return message


def get_O(umbrella):    # Returns the correct observation matrix given the current evidence
    if umbrella:
        return O_true
    else:
        return O_false


viterbi(5)
