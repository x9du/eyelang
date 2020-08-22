import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import image
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

classes = ('left', 'right', 'up', 'down', 'center')
classes_dict = {'left':0, 'right':1, 'up':2, 'down':3, 'center':4}

# define NN
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 7, padding=3) # change 1 to 3 channels?
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 32, 5, padding=2)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 4 * 7, 150)
        self.fc2 = nn.Linear(150, 80)
        self.fc3 = nn.Linear(80, 5) # output layer

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x.float())))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x) # output layer
        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:] # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

net = Net()
net.load_state_dict(torch.load('eyelang\\models\\eye-gaze_net_0.pth'))
transform = transforms.Compose([transforms.Resize((36, 60)), transforms.Grayscale(), transforms.ToTensor()])

fig = plt.figure(figsize=(15, 5))
for i in range(5):
    input = Image.open('eyelang\\cindy-eye\\cindy-%d.jpg' % (i))
    input = transform(input).numpy()
    input = (input - input.min()) / (input.max() - input.min()) * 255
    input = np.array([input])

    output = net(torch.from_numpy(input))
    _, predicted = torch.max(output.data, 1)

    fig.add_subplot(1, 5, i + 1)
    plt.imshow(input[0][0])
    print(classes[predicted[0]])
plt.show()