import argparse

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

from collections import OrderedDict

from PIL import Image
import numpy as np
import pandas as pd

import json


parser = argparse.ArgumentParser(description = "Train a model")
parser.add_argument('--data_dir', default = 'flowers')
parser.add_argument('--save_dir', default = '/home/workspace/ImageClassifier')
parser.add_argument('--arch', default = 'vgg19')
parser.add_argument('--learning_rate', default = 0.01)
parser.add_argument('--hidden_units', type = int,default = 512)
parser.add_argument('--epochs', default = 1)
parser.add_argument('--gpu', default = True)
args = parser.parse_args()



data_dir = args.data_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

train_transforms = transforms.Compose([transforms.RandomRotation(45),
                                      transforms.RandomVerticalFlip(30),
                                      transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])


test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])


train_dataset = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
valid_dataset = datasets.ImageFolder(data_dir + '/valid', transform=test_transforms)
test_dataset = datasets.ImageFolder(data_dir + '/test', transform=test_transforms)

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=64, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)



with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    
string = "model = models." + args.arch + "(pretrained=True)"
    
exec(string)

print(model)

for param in model.parameters():
    param.requires_grad = False
classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, 2048)),
                          ('relu', nn.ReLU()),
                          ('do',nn.Dropout(0.2)),
                          ('fc2', nn.Linear(2048, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    
model.classifier = classifier
print(model)

device = torch.device("cuda" if args.gpu else "cpu")



criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)
model.to(device);

epochs = args.epochs
steps = 0
running_loss = 0
print_every = 5
for epoch in range(epochs):
    for inputs, labels in train_dataloader:
        steps += 1
            
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        if steps % print_every == 0:
            test_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in valid_dataloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    
                    test_loss += batch_loss.item()
                    
                    
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Test loss: {test_loss/len(valid_dataloader):.3f}.. "
                  f"Test accuracy: {accuracy/len(valid_dataloader):.3f}")
            running_loss = 0
            model.train()
            
model.eval()
accuracy = 0
total_accuracy = 0
acc = 0
with torch.no_grad():
    for i,(inputs, labels) in enumerate(test_dataloader):
        inputs, labels = inputs.to(device), labels.to(device)
        logps = model.forward(inputs)
        batch_loss = criterion(logps, labels)
  
        ps = torch.exp(logps)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy = torch.mean(equals.type(torch.FloatTensor)).item()
        total_accuracy += accuracy
        acc = total_accuracy/(i+1)
    print(f"Validation accuracy: {acc:.3f}")
    
strings = "checkpoint = {'input_size': 59536,'output_size': 102,'classifier': model.classifier,'state_dict': model.state_dict(),'map' : train_dataset.class_to_idx,'model': models." + args.arch + "(pretrained=True)}"

exec(strings)
    
#string = "models." + args.arch + "(pretrained=True)"
    
    
#checkpoint = {'input_size': 59536,
              #'output_size': 102,
              #'classifier': model.classifier,
              #'state_dict': model.state_dict(),
              #'map' : train_dataset.class_to_idx,
              #'model': exec(string)}

torch.save(checkpoint, args.save_dir + '/checkpoint.pth')
