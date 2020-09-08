import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import MinMaxScaler
from random import random, seed
import json

print("Imported libraries")

# Column names gotten from https://archive.ics.uci.edu/ml/datasets/seeds#)
column_names = ["Area", "Perimeter", "Compactness", "Length of kernel", "Width of kernel", "Asymmetry coefficient", "Length of kernel groove", "Type"]

df = pd.read_csv(r"D:\GitRepos\course-machine-learning\week-three\AS10\seeds_dataset.csv", header=1, names=column_names)
print("Loaded dataframe")

dict_to_replace = {1:np.int(0), 2:np.int(1), 3:np.int(2)}
y = df["Type"]
y = y.replace(dict_to_replace)

X = df.drop("Type", axis=1)
scaler = MinMaxScaler() 
X = scaler.fit_transform(X)
X = pd.DataFrame(data=X)
print("Transformed features")

data = pd.concat((X, y),axis=1)
data = data.to_numpy()
print("Concat dataframes")


train, test = tts(data, train_size=0.8, random_state=2)

print(type(train))
print(type(test))
print(train.shape)
print(test.shape)

train = train.tolist()
test = test.tolist()

for row in train:
    row[-1] = int(row[-1])

for row in test:
    row[-1] = int(row[-1])

print(test[0:3])
print("Created train and test data")

def print_json_dump(dump):
    print(json.dumps(dump, sort_keys=True, indent=4))

# generate weights and bias
def create_network(n_inputs:int, n_hidden_layers:int, n_neurons_for_layer:list, n_outputs:int):
    """Creates a neural network with layers, neurons with weights and bias, output neurons with weights and bias

    Args:
        n_inputs (int): The amount of input features
        n_hidden_layers (int): The amount of hidden layers desired
        n_neurons_for_layer (list): A list containing the number of neurons per hidden layer
        n_outputs (int): Amount of output neurons wanted

    Returns:
        (dict): Your neural network
    """

    assert len(n_neurons_for_layer) == n_hidden_layers, \
        ("The length of this list needs to be the same as n_hidden_layers")

    network = []
    current_layer = -1

    for _ in range(n_hidden_layers):
        current_layer += 1
        layer = []
        for _ in range(n_neurons_for_layer[current_layer]):
            if current_layer == 0:
                weights = [random() for i in range(n_inputs)]
            elif current_layer > 0:
                weights = [random() for i in range(n_neurons_for_layer[current_layer-1])]

            bias = random()
            node = {"weights":weights, "bias":bias}
            layer.append(node)

        network.append(layer)

    
    n_output_weights = len(network[-1])
    layer = []
    for _ in range(n_outputs):
        weights = [random() for k in range(n_output_weights)]
        bias = random()
        node = {"weights":weights, "bias":bias}
        layer.append(node)

    network.append(layer)

    return network

# Neuron activation using sigmoid function
def sigmoid(x):
    val = 1/(1+np.exp(-x)) #maybe replace with math.exp
    return val

# Calculate the derivative of a neuron output
def sigmoid_deriv(x):
    return sigmoid(x)*(1-sigmoid(x))

# Calculate neuron activation for in input
def activate(weights, bias, inputs):
    activation = bias
    for i in range(len(weights)):
        activation += weights[i] * inputs[i]
    return activation

# Forward propagate input to a network output
def forward_propagate(network, row):
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = activate(neuron["weights"], neuron["bias"], inputs)
            neuron["output"] = sigmoid(activation)
            new_inputs.append(neuron["output"])
        inputs = new_inputs
    return inputs

def backward_propagate_error(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        if i != len(network)-1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i + 1]:
                    error += (neuron['weights'][j] * neuron['delta'])
                    errors.append(error)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(expected[j] - neuron['output'])
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * sigmoid_deriv(neuron['output'])

# Update weights with error
def update_weights(network, row, l_rate):
    for i in range(len(network)):
        inputs = row[:-1]
        if i != 0:
            inputs = [neuron["output"] for neuron in network[i - 1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron["weights"][j] += l_rate * neuron["delta"] * inputs[j]
            neuron["bias"] += l_rate * neuron["delta"]

# Train a network for a fixed number of epochs
def train_network(network, train, l_rate, n_epoch, n_outputs, n_hidden_layers, n_neurons_per_layer):
    for epoch in range(n_epoch):
        sum_error = 0
        for row in train:
            outputs = forward_propagate(network, row)
            expected = [0 for i in range(n_outputs)]
            expected[int(row[-1])] = 1
            sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
            backward_propagate_error(network, expected)
            update_weights(network, row, l_rate)
        print(f'>epoch={epoch}, lrate={l_rate}, error={sum_error}, hidden_layers={n_hidden_layers}, neurons/layer={n_neurons_per_layer}')

# Make a prediction with a network
def predict(network, row):
    outputs = forward_propagate(network, row)
    return outputs.index(max(outputs))

# Calculate accuracy
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0

# Backpropagation Algorithm with stochastic gradient descent
def back_propagation(train, l_rate, n_epoch, n_hidden_layers, n_neurons_per_layer):
    n_inputs = len(train[0]) - 1
    n_outputs = len(set([row[-1] for row in train]))
    network = create_network(n_inputs, n_hidden_layers, n_neurons_per_layer, n_outputs)
    train_network(network, train, l_rate, n_epoch, n_outputs, n_hidden_layers, n_neurons_per_layer)
    return(network)

print("Defined functions")


l_rate = 0.05
n_epoch = 10000
n_hidden_layers = 1
n_neurons_per_layer = [20]

# Train model
Model = back_propagation(train, l_rate, n_epoch, n_hidden_layers, n_neurons_per_layer)

# Make predictions on the test set
PredClass = list()
ActualClass = list()
for row in test:
    prediction = predict(Model, row)
    PredClass.append(prediction)
    ActualClass.append(row[-1])
    print('Expected=%d, Got=%d' % (row[-1], prediction))

accuracy = accuracy_metric(ActualClass, PredClass)
print("Accuracy:", accuracy)