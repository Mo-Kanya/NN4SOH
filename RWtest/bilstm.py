import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable


# feed in tensor data with shape ([batch_size, sequence_length, input_features])
class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # 2 for bidirection

    def forward(self, x):
        # Set initial states
        h0 = Variable(torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size)).cuda()  # 2 for bidirection
        c0 = Variable(torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size)).cuda()

        # Forward propagate RNN
        out, _ = self.lstm(x, (h0, c0))

        # Decode hidden state of last time step
        out = self.fc(out[:, -1, :])
        return out


"""Do not run it if not testing"""
# Hyper Parameters
if __name__ == '__main__':
    sequence_length = 28
    input_size = 28
    hidden_size = 128
    num_layers = 2
    num_classes = 10
    batch_size = 100
    num_epochs = 1
    learning_rate = 0.003
    # MNIST Dataset
    train_dataset = dsets.MNIST(root='./data/',
                                train=True,
                                transform=transforms.ToTensor(),
                                download=True)
    test_dataset = dsets.MNIST(root='./data/',
                               train=False,
                               transform=transforms.ToTensor())
    # Data Loader (Input Pipeline)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)
    rnn = BiLSTM(input_size, hidden_size, num_layers, num_classes)
    # rnn.cuda()

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)

    # Train the Model
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            print(images.shape)
            print(type(images))
            images = Variable(images.view(-1, sequence_length, input_size))  # .cuda()
            labels = Variable(labels)  # .cuda()
            print(images.shape)
            print(type(images))
            break

            # Forward + Backward + Optimize
            optimizer.zero_grad()
            outputs = rnn(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
                      % (epoch + 1, num_epochs, i + 1, len(train_dataset) // batch_size, loss.data.item()))
