import torch
import torch.nn.functional as F
from torch import nn


class Network(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, drop_p=0.5):
        super().__init__()
        
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])
        layers_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layers_sizes])
        self.output = nn.Linear(hidden_layers[-1], output_size)
        
        self.dropout = nn.Dropout(p=drop_p)
        
    def forward(self, x):
        for each in self.hidden_layers:
            x = F.relu(each(x))
            x = self.dropout(x)
        x = self.output(x)
        
        return F.log_softmax(x, dim=1)
    
def validation(model, testloader, criterion):
    accuracy = 0
    test_loss = 0
    for images, labels in testloader:
        images = images.view(images.shape[0], -1)
        output = model.forward(images)
        loss = criterion(output, labels)
        test_loss+=loss.item()
        
        ps = torch.exp(output)
        equality = (labels.data == ps.max(1)[1])
        accuracy+= equality.type_as(torch.FloatTensor()).mean()
        
    return test_loss, accuracy

def train(model, trainloader, testloader, criterion, optimizer, epochs=5, print_every=40):
    steps = 0
    running_loss = 0
    for e in range(epochs):
        model.train()
        for images, labels in trainloader:
            steps+=1
            images = images.view(images.shape[0], -1)
            
            optimizer.zero_grad()
            
            output = model.forward(images)
            
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            
            running_loss+=loss.item()
            
            if steps % print_every == 0:
                model.eval()
                
                with torch.no_grad():
                    test_loss, accuracy = validation(model, testloader, criterion)
                    
                print("Epoch: {}/{}.. ".format(e+1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                      "Test Loss: {:.3f}.. ".format(test_loss/len(testloader)),
                      "Test Accuracy: {:.3f}".format(accuracy/len(testloader)))

                running_loss = 0
                
                model.train()