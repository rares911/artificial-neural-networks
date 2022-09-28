import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error


#####################################################################################################################
#####################################################################################################################


#####################################################################################################################
#####################################################################################################################
# TODO - Application 3 - Step 5 - Create the ANN model
def modelDefinition():
    # TODO - Application 3 - Step 5a - Define the model as a Sequential model
    model = Sequential()

    # TODO - Application 3 - Step 5b - Add a Dense layer with 8 neurons to the model
    model.add(Dense(8, input_dim=13, kernel_initializer='normal', activation='relu'))
    model.add(Dense(16, input_dim=13, kernel_initializer='normal', activation='relu'))
    # TODO - Application 3 - Step 5c - Add a Dense layer (output layer) with 1 neuron
    model.add(Dense(1, kernel_initializer='normal'))
    model.add(Dense(1, kernel_initializer='normal'))
    # TODO - Application 3 - Step 5d - Compile the model by choosing the optimizer(adam) ant the loss function (MSE)
    model.compile(loss="mean_squared_error", optimizer="adam")
    return model


#####################################################################################################################
#####################################################################################################################


#####################################################################################################################
#####################################################################################################################
def main():
    # TODO - Application 3 - Step 1 - Read data from "Houses.csv" file
    csvFile = pd.read_csv("./Houses.csv").values

    # TODO - Application 3 - Step 2 - Shuffle the data
    np.random.shuffle(csvFile)

    # TODO - Application 3 - Step 3 - Separate the data from the labels (x_data / y_data)
    x_data = csvFile[:, 0:13]
    y_data = csvFile[:, 13:14]

    # TODO - Application 3 - Step 4 - Separate the data into training/testing dataset
    x_train = x_data[0:int(0.8 * len(x_data))]
    y_train = y_data[0:int(0.8 * len(y_data))]

    x_test = x_data[int(0.8 * len(x_data)):len(x_data)]
    y_test = y_data[int(0.8 * len(y_data)):len(y_data)]

    # TODO - Application 3 - Step 5 - Call the function "modelDefinition"
    predictedLabel = modelDefinition()
    # TODO - Application 3 - Step 6 - Train the model for 100 epochs and a batch of 16 samples
    predictedLabel.fit(x_train, y_train, epochs=100, batch_size=16, verbose=2)
    # TODO - Application 3 - Step 7 - Predict the house price for all the samples in the testing dataset
    predictions = predictedLabel.predict(x_test)
    # TODO - Exercise 8 - Compute the MSE for the test data
    # Computed MSE for the test DATA
    mse = np.square(np.subtract(x_test, predictions)).mean()
    print("Mean Square Error for test DATA = {}".format(mse))
    # TODO - Exercise 9
    # Valorile cand sunt citite nu au aceeasi pozitie si se foloseste din aplicatia 3, pasul 2 np.random.shuffle

    return


#####################################################################################################################
#####################################################################################################################


#####################################################################################################################
#####################################################################################################################
if __name__ == '__main__':
    main()
#####################################################################################################################
#####################################################################################################################
