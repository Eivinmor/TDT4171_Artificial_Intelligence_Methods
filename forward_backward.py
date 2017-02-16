import numpy as np

T = np.matrix([[0.7, 0.3],
               [0.3, 0.7]])

O_true = np.matrix([[0.9, 0],
                    [0, 0.2]])

O_false = np.matrix([[0.1, 0],
                     [0, 0.8]])

ev = [None, 1, 1, 0, 1, 1]


# def forward(prev_msg, evidence):
#     prediction = get_O(evidence) * np.transpose(T) * prev_msg
#     print("Prediction:\n", prediction)
#     normalisation = 1/prediction.sum()
#     print("Normalisation: ", normalisation)
#     prediction *= normalisation
#     print("\nFinal prediction:\n", prediction, "\n---------------")
#     return prediction
def forward(prev_msg, evidence):
    prediction = get_O(evidence) * np.transpose(T) * prev_msg
    return prediction / prediction.sum()


def estimate(t):
    return t


def forward_algorithm(t):
    t += 1
    fv = [0]*t
    fv[0] = np.matrix([[0.5], [0.5]])
    for i in range(1, t):
        print("\n\nIteration ", i, "\n---------------")
        fv[i] = forward(fv[i-1], ev[i])
        print(fv[i])


# noinspection PyPep8Naming
def get_O(umbrella):
    if umbrella:
        return O_true
    else:
        return O_false


forward_algorithm(3)
