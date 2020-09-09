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
def train_network(network, train, l_rate, n_epoch, n_outputs, n_neurons_per_layer):
    for epoch in range(n_epoch):
        sum_error = 0
        for row in train:
            outputs = forward_propagate(network, row)
            expected = [0 for i in range(n_outputs)]
            expected[int(row[-1])] = 1
            sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
            backward_propagate_error(network, expected)
            update_weights(network, row, l_rate)
        # print(f'>epoch={epoch}, lrate={l_rate}, error={sum_error}, hidden_layers={n_hidden_layers}, neurons/layer={n_neurons_per_layer}')
        print('>epoch=%i, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error), end="")
        print(", hidden_n=", n_neurons_per_layer)

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
    train_network(network, train, l_rate, n_epoch, n_outputs, n_neurons_per_layer)
    return(network)

print("Defined functions")

seed(1)

l_rate = 0.1
n_epoch = 500
n_hidden_layers = 2
# n_neurons_per_layer = [10]

all_accs = []
acc_sets = []

for x in range(1, 21):
    temp = []
    for y in range(1, 21):

        # Train model
        Model = back_propagation(train, l_rate, n_epoch, n_hidden_layers, [x, y])

        # Make predictions on the test set
        PredClass = list()
        ActualClass = list()
        for row in test:
            prediction = predict(Model, row)
            PredClass.append(prediction)
            ActualClass.append(row[-1])
            print('Expected=%d, Got=%d' % (row[-1], prediction))

        accuracy = accuracy_metric(ActualClass, PredClass)
        all_accs.append(accuracy)
        temp.append(accuracy)
        print("Hidden Layers:", n_hidden_layers)
        print("Neurons/Layer:", x)
        print("Accuracy:", accuracy)
        print(" ")
    acc_sets.append(temp)

print(all_accs)
print(acc_sets)

### Results of 
# l_rate = 0.1
# n_epoch = 500
# n_hidden_layers = 1
# neurons 1-20

# Neurons: 1 @ 16.666666666666664% accuracy
# Neurons: 2 @ 16.666666666666664% accuracy
# Neurons: 3 @ 90.47619047619048% accuracy
# Neurons: 4 @ 80.95238095238095% accuracy
# Neurons: 5 @ 85.71428571428571% accuracy
# Neurons: 6 @ 35.714285714285715% accuracy
# Neurons: 7 @ 90.47619047619048% accuracy
# Neurons: 8 @ 16.666666666666664% accuracy
# Neurons: 9 @ 80.95238095238095% accuracy
# Neurons: 10 @ 83.33333333333334% accuracy
# Neurons: 11 @ 85.71428571428571% accuracy
# Neurons: 12 @ 78.57142857142857% accuracy
# Neurons: 13 @ 90.47619047619048% accuracy
# Neurons: 14 @ 80.95238095238095% accuracy
# Neurons: 15 @ 88.09523809523809% accuracy
# Neurons: 16 @ 88.09523809523809% accuracy
# Neurons: 17 @ 92.85714285714286% accuracy
# Neurons: 18 @ 85.71428571428571% accuracy
# Neurons: 19 @ 88.09523809523809% accuracy
# Neurons: 20 @ 90.47619047619048% accuracy
#[16.666666666666664, 16.666666666666664, 90.47619047619048, 80.95238095238095, 85.71428571428571, 35.714285714285715, 90.47619047619048, 16.666666666666664, 80.95238095238095, 83.33333333333334, 85.71428571428571, 78.57142857142857, 90.47619047619048, 80.95238095238095, 88.09523809523809, 88.09523809523809, 92.85714285714286, 85.71428571428571, 88.09523809523809, 90.47619047619048]

### Results of 
# l_rate = 0.1
# n_epoch = 500
# n_hidden_layers = 2
# neurons 1-20 in each layer

# [16.666666666666664, 40.476190476190474, 47.61904761904761, 16.666666666666664, 42.857142857142854, 59.523809523809526, 26.190476190476193, 23.809523809523807, 16.666666666666664, 16.666666666666664, 59.523809523809526, 26.190476190476193, 33.33333333333333, 54.761904761904766, 57.14285714285714, 23.809523809523807, 69.04761904761905, 54.761904761904766, 16.666666666666664, 66.66666666666666, 28.57142857142857, 45.23809523809524, 40.476190476190474, 38.095238095238095, 42.857142857142854, 16.666666666666664, 61.904761904761905, 50.0, 50.0, 59.523809523809526, 59.523809523809526, 64.28571428571429, 59.523809523809526, 61.904761904761905, 64.28571428571429, 54.761904761904766, 45.23809523809524, 64.28571428571429, 61.904761904761905, 57.14285714285714, 16.666666666666664, 16.666666666666664, 40.476190476190474, 47.61904761904761, 47.61904761904761, 16.666666666666664, 23.809523809523807, 64.28571428571429, 66.66666666666666, 66.66666666666666, 59.523809523809526, 40.476190476190474, 59.523809523809526, 47.61904761904761, 52.38095238095239, 59.523809523809526, 66.66666666666666, 66.66666666666666, 59.523809523809526, 61.904761904761905, 59.523809523809526, 38.095238095238095, 59.523809523809526, 16.666666666666664, 40.476190476190474, 50.0, 47.61904761904761, 64.28571428571429, 71.42857142857143, 40.476190476190474, 71.42857142857143, 59.523809523809526, 47.61904761904761, 47.61904761904761, 69.04761904761905, 47.61904761904761, 47.61904761904761, 54.761904761904766, 64.28571428571429, 47.61904761904761, 16.666666666666664, 45.23809523809524, 45.23809523809524, 16.666666666666664, 16.666666666666664, 76.19047619047619, 69.04761904761905, 16.666666666666664, 59.523809523809526, 33.33333333333333, 47.61904761904761, 57.14285714285714, 69.04761904761905, 66.66666666666666, 61.904761904761905, 59.523809523809526, 66.66666666666666, 47.61904761904761, 47.61904761904761, 71.42857142857143, 16.666666666666664, 16.666666666666664, 59.523809523809526, 16.666666666666664, 59.523809523809526, 16.666666666666664, 54.761904761904766, 19.047619047619047, 59.523809523809526, 64.28571428571429, 47.61904761904761, 61.904761904761905, 47.61904761904761, 33.33333333333333, 16.666666666666664, 73.80952380952381, 71.42857142857143, 47.61904761904761, 64.28571428571429, 64.28571428571429, 16.666666666666664, 59.523809523809526, 33.33333333333333, 66.66666666666666, 16.666666666666664, 38.095238095238095, 26.190476190476193, 64.28571428571429, 26.190476190476193, 54.761904761904766, 40.476190476190474, 61.904761904761905, 47.61904761904761, 47.61904761904761, 50.0, 73.80952380952381, 73.80952380952381, 57.14285714285714, 50.0, 47.61904761904761, 16.666666666666664, 54.761904761904766, 64.28571428571429, 61.904761904761905, 54.761904761904766, 50.0, 66.66666666666666, 73.80952380952381, 47.61904761904761, 66.66666666666666, 57.14285714285714, 71.42857142857143, 59.523809523809526, 69.04761904761905, 47.61904761904761, 57.14285714285714, 47.61904761904761, 47.61904761904761, 64.28571428571429, 59.523809523809526, 59.523809523809526, 50.0, 47.61904761904761, 59.523809523809526, 57.14285714285714, 57.14285714285714, 47.61904761904761, 16.666666666666664, 38.095238095238095, 73.80952380952381, 47.61904761904761, 47.61904761904761, 61.904761904761905, 54.761904761904766, 47.61904761904761, 50.0, 64.28571428571429, 47.61904761904761, 59.523809523809526, 59.523809523809526, 16.666666666666664, 16.666666666666664, 54.761904761904766, 16.666666666666664, 16.666666666666664, 16.666666666666664, 47.61904761904761, 66.66666666666666, 47.61904761904761, 47.61904761904761, 16.666666666666664, 69.04761904761905, 47.61904761904761, 64.28571428571429, 54.761904761904766, 47.61904761904761, 61.904761904761905, 64.28571428571429, 69.04761904761905, 61.904761904761905, 59.523809523809526, 16.666666666666664, 57.14285714285714, 54.761904761904766, 69.04761904761905, 21.428571428571427, 47.61904761904761, 71.42857142857143, 73.80952380952381, 59.523809523809526, 59.523809523809526, 47.61904761904761, 47.61904761904761, 59.523809523809526, 47.61904761904761, 47.61904761904761, 47.61904761904761, 47.61904761904761, 54.761904761904766, 57.14285714285714, 61.904761904761905, 16.666666666666664, 16.666666666666664, 59.523809523809526, 47.61904761904761, 57.14285714285714, 47.61904761904761, 73.80952380952381, 88.09523809523809, 54.761904761904766, 54.761904761904766, 16.666666666666664, 71.42857142857143, 59.523809523809526, 57.14285714285714, 52.38095238095239, 23.809523809523807, 47.61904761904761, 61.904761904761905, 47.61904761904761, 16.666666666666664, 16.666666666666664, 61.904761904761905, 16.666666666666664, 71.42857142857143, 35.714285714285715, 52.38095238095239, 69.04761904761905, 38.095238095238095, 59.523809523809526, 16.666666666666664, 47.61904761904761, 47.61904761904761, 23.809523809523807, 47.61904761904761, 47.61904761904761, 54.761904761904766, 59.523809523809526, 69.04761904761905, 59.523809523809526, 16.666666666666664, 52.38095238095239, 16.666666666666664, 21.428571428571427, 80.95238095238095, 16.666666666666664, 61.904761904761905, 83.33333333333334, 64.28571428571429, 47.61904761904761, 47.61904761904761, 71.42857142857143, 64.28571428571429, 59.523809523809526, 71.42857142857143, 66.66666666666666, 50.0, 85.71428571428571, 57.14285714285714, 47.61904761904761, 16.666666666666664, 64.28571428571429, 59.523809523809526, 50.0, 76.19047619047619, 61.904761904761905, 28.57142857142857, 16.666666666666664, 66.66666666666666, 69.04761904761905, 57.14285714285714, 16.666666666666664, 26.190476190476193, 54.761904761904766, 64.28571428571429, 47.61904761904761, 47.61904761904761, 16.666666666666664, 47.61904761904761, 47.61904761904761, 16.666666666666664, 61.904761904761905, 61.904761904761905, 16.666666666666664, 38.095238095238095, 54.761904761904766, 47.61904761904761, 16.666666666666664, 47.61904761904761, 47.61904761904761, 64.28571428571429, 26.190476190476193, 64.28571428571429, 59.523809523809526, 47.61904761904761, 57.14285714285714, 47.61904761904761, 47.61904761904761, 47.61904761904761, 47.61904761904761, 16.666666666666664, 16.666666666666664, 57.14285714285714, 16.666666666666664, 61.904761904761905, 16.666666666666664, 59.523809523809526, 76.19047619047619, 16.666666666666664, 47.61904761904761, 47.61904761904761, 69.04761904761905, 47.61904761904761, 83.33333333333334, 50.0, 52.38095238095239, 59.523809523809526, 69.04761904761905, 57.14285714285714, 57.14285714285714, 16.666666666666664, 16.666666666666664, 16.666666666666664, 88.09523809523809, 40.476190476190474, 16.666666666666664, 47.61904761904761, 47.61904761904761, 47.61904761904761, 47.61904761904761, 57.14285714285714, 47.61904761904761, 85.71428571428571, 16.666666666666664, 47.61904761904761, 42.857142857142854, 59.523809523809526, 57.14285714285714, 47.61904761904761, 47.61904761904761, 61.904761904761905, 16.666666666666664, 16.666666666666664, 16.666666666666664, 69.04761904761905, 66.66666666666666, 16.666666666666664, 59.523809523809526, 50.0, 61.904761904761905, 47.61904761904761, 61.904761904761905, 47.61904761904761, 47.61904761904761, 47.61904761904761, 71.42857142857143, 47.61904761904761, 47.61904761904761, 54.761904761904766, 50.0, 16.666666666666664, 59.523809523809526, 16.666666666666664, 80.95238095238095, 80.95238095238095, 78.57142857142857, 52.38095238095239, 50.0, 69.04761904761905, 45.23809523809524, 47.61904761904761, 16.666666666666664, 47.61904761904761, 71.42857142857143, 47.61904761904761, 47.61904761904761, 52.38095238095239, 64.28571428571429, 71.42857142857143, 47.61904761904761]
# [[16.666666666666664, 40.476190476190474, 47.61904761904761, 16.666666666666664, 42.857142857142854, 59.523809523809526, 26.190476190476193, 23.809523809523807, 16.666666666666664, 16.666666666666664, 59.523809523809526, 26.190476190476193, 33.33333333333333, 54.761904761904766, 57.14285714285714, 23.809523809523807, 69.04761904761905, 54.761904761904766, 16.666666666666664, 66.66666666666666], [28.57142857142857, 45.23809523809524, 40.476190476190474, 38.095238095238095, 42.857142857142854, 16.666666666666664, 61.904761904761905, 50.0, 50.0, 59.523809523809526, 59.523809523809526, 64.28571428571429, 59.523809523809526, 61.904761904761905, 64.28571428571429, 54.761904761904766, 45.23809523809524, 64.28571428571429, 61.904761904761905, 57.14285714285714], [16.666666666666664, 16.666666666666664, 40.476190476190474, 47.61904761904761, 47.61904761904761, 16.666666666666664, 23.809523809523807, 64.28571428571429, 66.66666666666666, 66.66666666666666, 59.523809523809526, 40.476190476190474, 59.523809523809526, 47.61904761904761, 52.38095238095239, 59.523809523809526, 66.66666666666666, 66.66666666666666, 59.523809523809526, 61.904761904761905], [59.523809523809526, 38.095238095238095, 59.523809523809526, 16.666666666666664, 40.476190476190474, 50.0, 47.61904761904761, 64.28571428571429, 71.42857142857143, 40.476190476190474, 71.42857142857143, 59.523809523809526, 47.61904761904761, 47.61904761904761, 69.04761904761905, 47.61904761904761, 47.61904761904761, 54.761904761904766, 64.28571428571429, 47.61904761904761], [16.666666666666664, 45.23809523809524, 45.23809523809524, 16.666666666666664, 16.666666666666664, 76.19047619047619, 69.04761904761905, 16.666666666666664, 59.523809523809526, 33.33333333333333, 47.61904761904761, 57.14285714285714, 69.04761904761905, 66.66666666666666, 61.904761904761905, 59.523809523809526, 66.66666666666666, 47.61904761904761, 47.61904761904761, 71.42857142857143], [16.666666666666664, 16.666666666666664, 59.523809523809526, 16.666666666666664, 59.523809523809526, 16.666666666666664, 54.761904761904766, 19.047619047619047, 59.523809523809526, 64.28571428571429, 47.61904761904761, 61.904761904761905, 47.61904761904761, 33.33333333333333, 16.666666666666664, 73.80952380952381, 71.42857142857143, 47.61904761904761, 64.28571428571429, 64.28571428571429], [16.666666666666664, 59.523809523809526, 33.33333333333333, 66.66666666666666, 16.666666666666664, 38.095238095238095, 26.190476190476193, 64.28571428571429, 26.190476190476193, 54.761904761904766, 40.476190476190474, 61.904761904761905, 47.61904761904761, 47.61904761904761, 50.0, 73.80952380952381, 73.80952380952381, 57.14285714285714, 50.0, 47.61904761904761], [16.666666666666664, 54.761904761904766, 64.28571428571429, 61.904761904761905, 54.761904761904766, 50.0, 66.66666666666666, 73.80952380952381, 47.61904761904761, 66.66666666666666, 57.14285714285714, 71.42857142857143, 59.523809523809526, 69.04761904761905, 47.61904761904761, 57.14285714285714, 47.61904761904761, 47.61904761904761, 64.28571428571429, 59.523809523809526], [59.523809523809526, 50.0, 47.61904761904761, 59.523809523809526, 57.14285714285714, 57.14285714285714, 47.61904761904761, 16.666666666666664, 38.095238095238095, 73.80952380952381, 47.61904761904761, 47.61904761904761, 61.904761904761905, 54.761904761904766, 47.61904761904761, 50.0, 64.28571428571429, 47.61904761904761, 59.523809523809526, 59.523809523809526], [16.666666666666664, 16.666666666666664, 54.761904761904766, 16.666666666666664, 16.666666666666664, 16.666666666666664, 47.61904761904761, 66.66666666666666, 47.61904761904761, 47.61904761904761, 16.666666666666664, 69.04761904761905, 47.61904761904761, 64.28571428571429, 54.761904761904766, 47.61904761904761, 61.904761904761905, 64.28571428571429, 69.04761904761905, 61.904761904761905], [59.523809523809526, 16.666666666666664, 57.14285714285714, 54.761904761904766, 69.04761904761905, 21.428571428571427, 47.61904761904761, 71.42857142857143, 73.80952380952381, 59.523809523809526, 59.523809523809526, 47.61904761904761, 47.61904761904761, 59.523809523809526, 47.61904761904761, 47.61904761904761, 47.61904761904761, 47.61904761904761, 54.761904761904766, 57.14285714285714], [61.904761904761905, 16.666666666666664, 16.666666666666664, 59.523809523809526, 47.61904761904761, 57.14285714285714, 47.61904761904761, 73.80952380952381, 88.09523809523809, 54.761904761904766, 54.761904761904766, 16.666666666666664, 71.42857142857143, 59.523809523809526, 57.14285714285714, 52.38095238095239, 23.809523809523807, 47.61904761904761, 61.904761904761905, 47.61904761904761], [16.666666666666664, 16.666666666666664, 61.904761904761905, 16.666666666666664, 71.42857142857143, 35.714285714285715, 52.38095238095239, 69.04761904761905, 38.095238095238095, 59.523809523809526, 16.666666666666664, 47.61904761904761, 47.61904761904761, 23.809523809523807, 47.61904761904761, 47.61904761904761, 54.761904761904766, 59.523809523809526, 69.04761904761905, 59.523809523809526], [16.666666666666664, 52.38095238095239, 16.666666666666664, 21.428571428571427, 80.95238095238095, 16.666666666666664, 61.904761904761905, 83.33333333333334, 64.28571428571429, 47.61904761904761, 47.61904761904761, 71.42857142857143, 64.28571428571429, 59.523809523809526, 71.42857142857143, 66.66666666666666, 50.0, 85.71428571428571, 57.14285714285714, 47.61904761904761], [16.666666666666664, 64.28571428571429, 59.523809523809526, 50.0, 76.19047619047619, 61.904761904761905, 28.57142857142857, 16.666666666666664, 66.66666666666666, 69.04761904761905, 57.14285714285714, 16.666666666666664, 26.190476190476193, 54.761904761904766, 64.28571428571429, 47.61904761904761, 47.61904761904761, 16.666666666666664, 47.61904761904761, 47.61904761904761], [16.666666666666664, 61.904761904761905, 61.904761904761905, 16.666666666666664, 38.095238095238095, 54.761904761904766, 47.61904761904761, 16.666666666666664, 47.61904761904761, 47.61904761904761, 64.28571428571429, 26.190476190476193, 64.28571428571429, 59.523809523809526, 47.61904761904761, 57.14285714285714, 47.61904761904761, 47.61904761904761, 47.61904761904761, 47.61904761904761], [16.666666666666664, 16.666666666666664, 57.14285714285714, 16.666666666666664, 61.904761904761905, 16.666666666666664, 59.523809523809526, 76.19047619047619, 16.666666666666664, 47.61904761904761, 47.61904761904761, 69.04761904761905, 47.61904761904761, 83.33333333333334, 50.0, 52.38095238095239, 59.523809523809526, 69.04761904761905, 57.14285714285714, 57.14285714285714], [16.666666666666664, 16.666666666666664, 16.666666666666664, 88.09523809523809, 40.476190476190474, 16.666666666666664, 47.61904761904761, 47.61904761904761, 47.61904761904761, 47.61904761904761, 57.14285714285714, 47.61904761904761, 85.71428571428571, 16.666666666666664, 47.61904761904761, 42.857142857142854, 59.523809523809526, 57.14285714285714, 47.61904761904761, 47.61904761904761], [61.904761904761905, 16.666666666666664, 16.666666666666664, 16.666666666666664, 69.04761904761905, 66.66666666666666, 16.666666666666664, 59.523809523809526, 50.0, 61.904761904761905, 47.61904761904761, 61.904761904761905, 47.61904761904761, 47.61904761904761, 47.61904761904761, 71.42857142857143, 47.61904761904761, 47.61904761904761, 54.761904761904766, 50.0], [16.666666666666664, 59.523809523809526, 16.666666666666664, 80.95238095238095, 80.95238095238095, 78.57142857142857, 52.38095238095239, 50.0, 69.04761904761905, 45.23809523809524, 47.61904761904761, 16.666666666666664, 47.61904761904761, 71.42857142857143, 47.61904761904761, 47.61904761904761, 52.38095238095239, 64.28571428571429, 71.42857142857143, 47.61904761904761]]