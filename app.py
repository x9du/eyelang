import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import image
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import io
from flask import Flask, jsonify, request, make_response

app = Flask(__name__)

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
net.load_state_dict(torch.load('models\\eye-gaze_net_0.pth'))

@app.route('/', methods=['POST'])
def predict():
    if request.method == 'POST':
        data = request.form.to_dict()
        # print(data)
        
        imageArr = []
        imageLength = int(data['file[height]']) * int(data['file[width]'])
        for i in range(imageLength):
            imageArr.append((int(data['file[data][%d]' % (i * 4)]) + int(data['file[data][%d]' % (i * 4 + 1)]) + int(data['file[data][%d]' % (i * 4 + 2)])) / 3)
        # print(imageArr)
        
        class_name = get_prediction(imageArr, int(data['file[height]']))
        response = make_response(jsonify({'class': class_name}))
        response.headers['Access-Control-Allow-Origin'] = '*'
        return response

def transform_image(image_bytes, height):
    transform = transforms.Compose([transforms.Resize((36, 60)), transforms.ToTensor()])
    input = Image.fromarray(np.reshape(np.array(image_bytes), (height, -1)))
    input = transform(input).numpy()
    input = (input - input.min()) / (input.max() - input.min()) * 255
    input = np.array([input])
    input.setflags(write=True)
    input = torch.from_numpy(input)    
    return input

def get_prediction(image_bytes, height):
    output = net(transform_image(image_bytes, height))
    _, predicted = torch.max(output.data, 1)
    return classes[predicted[0]]

if __name__ == '__main__':
    app.run()