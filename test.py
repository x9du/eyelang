import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from train import Net

if __name__ == '__main__':
    torch.multiprocessing.freeze_support()

    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    PATH = './eyelang/cifar_net.pth'
    # test the NN on test data
    # re-load saved model
    net = Net()
    net.load_state_dict(torch.load(PATH))

    dataiter = iter(testloader)
    images, labels = dataiter.next()

    # print images
    # permute to have channels as last dimension
    plt.imshow(torchvision.utils.make_grid(images).permute(1, 2, 0))
    plt.show()
    print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))
    outputs = net(images) # neural net's classifications
    _, predicted = torch.max(outputs, 1) # get highest energy/most likely label
    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))