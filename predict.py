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



parser = argparse.ArgumentParser(description = "Prediction")
parser.add_argument('--checkpoint', default = '/home/workspace/ImageClassifier/checkpoint.pth')
parser.add_argument('--image_dir', default = '/home/workspace/ImageClassifier/flowers/test/1/image_06752.jpg')
parser.add_argument('--category_names', default = '/home/workspace/ImageClassifier/cat_to_name.json')
parser.add_argument('--top_k', default = 5)
parser.add_argument('--gpu', default = True)
args = parser.parse_args()



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




with open(args.category_names, 'r') as f:
    cat_to_name = json.load(f)

    
device = torch.device("cuda" if args.gpu else "cpu")

def load_fun(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    
    return model,checkpoint['map']

model, maps = load_fun(args.checkpoint)


def process_image(image):
    with Image.open(image) as im:
                           
        tensor_image = test_transforms(im)
        
        np_image = tensor_image.numpy()
        
        tensor_image = torch.from_numpy(np_image)
              
        return tensor_image
    
    
def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    images = process_image(image_path)
    images = images.unsqueeze(0)
    images = images.to(device)
    model = model.to(device)
    with torch.no_grad():
        logps = model.forward(images)
        probs = torch.exp(logps)
        top_probs, top_classes = probs.topk(topk)

        
        return top_probs, top_classes

    
probs, flowers = predict(args.image_dir, model,args.top_k)
inverted_dict = dict(map(reversed,maps.items()))
probabilities = []
for i in range(len(probs[0])):
    x = probs[0][i].item()
    probabilities.append(x)
indeces = []
results = []
for c in range(len(flowers[0])):
    x = inverted_dict[flowers[0][c].item()]
    indeces.append(x)
for i in indeces:
    x = cat_to_name[i]
    results.append(x)
print(probabilities)    
print(results)