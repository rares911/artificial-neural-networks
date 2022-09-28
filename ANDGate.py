import numpy as np
#####################################################################################################################
#####################################################################################################################


#####################################################################################################################
#####################################################################################################################
def activationFunction(n):
    # TODO - Application 1 - Step 4b - Define the binary step function as activation function
    if n>=0:
        return 1
    else:
        return 0


#####################################################################################################################
#####################################################################################################################


#####################################################################################################################
#####################################################################################################################
def forwardPropagation(p, weights, bias):
    a = None  # the neuron output

    # TODO - Application 1 - Step 4a - Multiply weights with the input vector (p) and add the bias   =>  n
    n = (weights[0] * p[0]) + (p[1] * weights[1]) + bias
    #n = np.dot(p, weights) + bias

    # TODO - Application 1 - Step 4c - Pass the result to the activation function  =>  a
    a = activationFunction(n)

    return a


#####################################################################################################################
#####################################################################################################################


#####################################################################################################################
#####################################################################################################################
def main():
    # Application 1 - Train a single neuron perceptron in order to predict the output of an AND gate.
    # The network should receive as input two values (0 or 1) and should predict the target output

    # Input data
    P = [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ]

    # Labels
    t = [0, 0, 0, 1]

    # TODO - Application 1 - Step 2 - Initialize the weights with zero  (weights)
    weights = [0, 0]

    # TODO - Application 1 - Step 2 - Initialize the bias with zero  (bias)
    bias = 0

    # TODO - Application 1 - Step 3 - Set the number of training steps  (epochs)
    epochs = 4

    # TODO - Application 1 - Step 4 - Perform the neuron training for multiple epochs
    for ep in range(epochs):
        for i in range(len(t)):
            # TODO - Application 1 - Step 4 - Call the forwardPropagation method
            predictedLabel = forwardPropagation(P[i], weights, bias)

            # TODO - Application 1 - Step 5 - Compute the prediction error (error)
            error = t[i] - predictedLabel

            # TODO - Application 1 - Step 6 - Update the weights
            weights[0] = weights[0] + error * P[i][0]
            weights[1] = weights[1] + error * P[i][1]

            # TODO - Application 1 - Step 7 - Update the bias
            bias = bias + error


    # TODO - Application 1 - Step 8 - Print weights and bias
    print('Weights = {}, Bias = {}'.format(weights, bias))

    # TODO - Application 1 - Step 9 - Display the results
    for idx, p in enumerate(P):
        predLab = forwardPropagation(p, weights, bias)
        print("Punctul {} are eticheta prezisa {} si eticheta corecta {} ".format(p, predLab, t[idx]))
    return


#####################################################################################################################
#####################################################################################################################


#####################################################################################################################
#####################################################################################################################
if __name__ == "__main__":
    main()
#####################################################################################################################
#####################################################################################################################
