import numpy as np

T = np.matrix([[0.7, 0.3],      # Transition model
               [0.3, 0.7]])

O_true = np.matrix([[0.9, 0],   # Observation model for U_t = true
                    [0, 0.2]])

O_false = np.matrix([[0.1, 0],  # Observation model for U_t = false
                     [0, 0.8]])

ev = [0, 0, 1, 1, 0]            # Evidence vector


def forward(prev_msg, evidence):  # Calculate forward message
    prediction = get_O(evidence) * np.transpose(T) * prev_msg   # O_t * T(transposed) * f_{t-1}
    normalisation = prediction.sum()                            # Calculate normalisation
    return prediction / normalisation                           # Return normalised values


def viterbi(t):
    # Initialisation
    mv = [None]*t   # List of message matrices in format [[R_t = true],[R_t = false]]
    edges = [[True for i in range(2)] for j in range(t-1)]  # Initialising array for storing edges
    mv[0] = forward(np.matrix([[0.5], [0.5]]), ev[0])       # Performing initial step (filtering)
    print("\n", mv[0])

    # Calculate sequence probabilities
    for i in range(1, t):   # Loop all intervals
        mv[i] = calc_viterbi_message(mv[i-1], ev[i], edges[i-1])    # Calculate current message
        print("\n", mv[i])

    # Backtrack path
    pathState = mv[-1][0] > mv[-1][1]                   # Get last state
    solution = [None]*t
    solution[t-1] = (bool(pathState))                   # Add last state to solution
    for i in range(2, t+1):
        pathState = edges[t-i][int(pathState)]          # Get previous state value in sequence
        solution[t-i] = pathState                       # Add path to solution in reverse
    return solution


def calc_viterbi_message(prev_msg, evidence, intervalEdges):
    path_probs = np.multiply(prev_msg.transpose(), T)   # Calculate probabilities of the paths
    max_path_probs = path_probs.max(1)                  # Select max path probabilities per current state (matrix row)
    message = get_O(evidence) * max_path_probs          # Multiply with observation probabilities
    if max_path_probs[0] == path_probs[0, 1]:           # Save edges that lead to most probable sequence for each state
        intervalEdges[1] = False
    if max_path_probs[1] == path_probs[1, 1]:
        intervalEdges[0] = False
    return message


def get_O(umbrella):    # Returns the correct observation matrix given the current evidence
    if umbrella:
        return O_true
    else:
        return O_false

print("Evidence: ", ev)
most_likely_sequence = viterbi(5)
print("\n", most_likely_sequence)
