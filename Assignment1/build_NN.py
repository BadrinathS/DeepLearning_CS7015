from numpy.core.numeric import cross
from numpy.lib.twodim_base import _trilu_indices_form_dispatcher
import torch
import torchvision
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from torchvision import transforms


def sigmoid(arr):
    return (1/(1+np.exp(-arr)))

def grad_sigmoid(arr):
    return sigmoid(arr)*(1-sigmoid(arr))

def softmax(arr):
    denom = np.sum(np.exp(arr), axis= 0)
    arr = np.exp(arr)
    return np.divide(arr, denom)

def one_hot_vector(label):
    arr = np.zeros((10,label.shape[0]))
    for i, val in enumerate(label):
        arr[label,i] = 1

    return arr

def initialise_gradients(hidden_layers, neurons):
    gradients = {}
    gradients["dh0"] = np.zeros((neurons[0],1))
    gradients["da0"] = np.zeros((neurons[0],1))
    for i in range(1, hidden_layers+2):
        gradients["dW"+str(i)] = np.zeros((neurons[i], neurons[i-1]))
        gradients["db"+str(i)] = np.zeros((neurons[i],1))
        gradients["da"+str(i)] = np.zeros((neurons[i],1))
        gradients["dh"+str(i)] = np.zeros((neurons[i],1))
    
    return gradients

def cross_entropy(X,Y):
    X = np.array(X)
    Y = np.array(Y)

    loss = np.zeros((1,X.shape[1]))
    for i in range(X.shape[1]):
        loss[0,i] = np.sum(-1*Y[:,i]*np.log(X[:,i]))
    
    return loss


def forward_pass(input, parameters, hidden_layers):

    input = input.reshape(input.shape[1]*input.shape[2]*input.shape[3], input.shape[0])

    H = {}
    A = {}
    H["h0"] = input
    
    for i in range(1, hidden_layers+1):
        A["a"+str(i)] = np.dot(parameters["W"+str(i)], H["h"+str(i-1)]) + parameters["b"+str(i)]
        H["h"+str(i)] = sigmoid(A["a"+str(i)])
    A["a"+str(hidden_layers+1)] = np.dot(parameters["W"+str(hidden_layers+1)], H["h"+str(hidden_layers)]) + parameters["b"+str(hidden_layers+1)]
    H["h"+str(hidden_layers+1)] = softmax(A["a"+str(hidden_layers+1)])


    return H["h"+str(hidden_layers+1)], A, H

def backpropagation(parameters, A, H, out, labels, loss, hidden_layers, neurons, batch_size):

    labels_vector = one_hot_vector(labels)
    A["a0"] = np.zeros((neurons[0], batch_size))

    gradients = initialise_gradients(hidden_layers, neurons)
    gradients["da"+str(hidden_layers+1)] = out - labels_vector

    for i in np.arange(hidden_layers+1, 0, -1):
        gradients["dW"+str(i)] = np.dot(gradients["da"+str(i)], H["h"+str(i-1)].T)
        gradients["db"+str(i)] = gradients["da"+str(i)]
        gradients["dh"+str(i-1)] = np.dot(parameters["W"+str(i)].T, gradients["da"+str(i)])
        gradients["da"+str(i-1)] = gradients["dh"+str(i-1)]*grad_sigmoid(A["a"+str(i-1)])

    return gradients


    


def main():

    batch_size = 1
    hidden_layers = 3

    neurons = [100, 100, 400]
    neurons = [28*28] + neurons + [10]
    
    parameters = {}

    for i in range(1, hidden_layers+2):
        parameters["W"+str(i)] = np.random.randn(neurons[i], neurons[i-1])
        parameters["b"+str(i)] = np.random.randn(neurons[i],1)




    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    trainset = torchvision.datasets.FashionMNIST('MNIST_data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    train_dataset_array = next(iter(trainloader))[0].numpy()
    # print(train_dataset_array.shape)

    testset = torchvision.datasets.FashionMNIST('MNIST_data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)
    test_dataset_array = next(iter(testloader))[0].numpy()

    # Feed forward code

    for i, batch in enumerate(iter(trainloader)):
        batch = next(iter(trainloader))
        images, labels = batch
        out, A, H = forward_pass(images, parameters, hidden_layers)
        # print(out.shape)
        loss = cross_entropy(out, one_hot_vector(labels.numpy()))
        gradients = backpropagation(parameters, A, H, out, labels, loss, hidden_layers, neurons, batch_size)
        print(gradients)
        exit()
        




if __name__ == "__main__":
    main()