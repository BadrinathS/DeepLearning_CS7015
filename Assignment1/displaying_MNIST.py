from numpy.lib.twodim_base import _trilu_indices_form_dispatcher
import torch
import torchvision
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from torchvision import transforms


def output_label(label):
    output_mapping = {
                 0: "T-shirt/Top",
                 1: "Trouser",
                 2: "Pullover",
                 3: "Dress",
                 4: "Coat", 
                 5: "Sandal", 
                 6: "Shirt",
                 7: "Sneaker",
                 8: "Bag",
                 9: "Ankle Boot"
                 }
    input = (label.item() if type(label) == torch.Tensor else label)
    return output_mapping[input]

batch_size = 10


transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

trainset = torchvision.datasets.FashionMNIST('MNIST_data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
train_dataset_array = next(iter(trainloader))[0].numpy()
# print(train_dataset_array.shape)

testset = torchvision.datasets.FashionMNIST('MNIST_data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)
test_dataset_array = next(iter(testloader))[0].numpy()
# print(test_dataset_array.shape)


batch = next(iter(trainloader))
images, labels = batch

grid = torchvision.utils.make_grid(images, nrow=10)

plt.figure()
for i, label in enumerate(labels):
    plt.subplot(1,batch_size,i+1)
    plt.imshow(255.0*((images.numpy()[0][0]/np.max(images.numpy()[0][0].reshape(28,28)))+ 1))
    plt.title(output_label(label))

plt.show()
