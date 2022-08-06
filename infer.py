# Mainly https://colab.research.google.com/drive/1Z1lbR_oTSaeodv9tTm11uEhOjhkUx1L4?usp=sharing#scrollTo=5ql2T5PDUI1D
# Original code by Roboflow team
# Modified by Walter Liu

import torch
import matplotlib.pyplot as plt
import cv2
import torchvision
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms
from transformers import ViTModel
from transformers.modeling_outputs import SequenceClassifierOutput
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTFeatureExtractor
import torch.nn as nn
import torch
import torch.utils.data as data
from torch.autograd import Variable
import numpy as np

print('Setup complete. Using torch %s %s' % (torch.__version__, torch.cuda.get_device_properties(0) if torch.cuda.is_available() else 'CPU'))

LEARNING_RATE = 2e-5
dataset_dir = "C:/Projects/Music_Classifier_Generator/dataset/"

train_ds = torchvision.datasets.ImageFolder(dataset_dir + 'train/', transform=ToTensor())
valid_ds = torchvision.datasets.ImageFolder(dataset_dir + 'valid/', transform=ToTensor())
test_ds = torchvision.datasets.ImageFolder(dataset_dir + 'test/', transform=ToTensor())

class ViTForImageClassification(nn.Module):
    def __init__(self, num_labels=3):
        super(ViTForImageClassification, self).__init__()
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.vit.config.hidden_size, num_labels)
        self.num_labels = num_labels

    def forward(self, pixel_values, labels):
        outputs = self.vit(pixel_values=pixel_values)
        output = self.dropout(outputs.last_hidden_state[:,0])
        logits = self.classifier(output)

        loss = None
        if labels is not None:
          loss_fct = nn.CrossEntropyLoss()
          loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        if loss is not None:
          return logits, loss.item()
        else:
          return logits, None
      
def infer(model_path, img_path):

    # Use GPU if available  
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

    model = torch.load(model_path, map_location=device)
    model.eval()

    # Feature Extractor
    feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
    # Adam Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # Cross Entropy Loss
    loss_func = nn.CrossEntropyLoss()
    
    if torch.cuda.is_available():
        model.cuda()
        
    # Disable grad
    with torch.no_grad():
        
      img = np.array(cv2.imread(img_path))
    
      target = torch.tensor([0])
    
      tens = torch.tensor(img)
    
      # print the converted image tensor
      inputs = tens
      # Reshape and get feature matrices as needed
      print(inputs.shape)
      # Save original Input
      originalInput = inputs
      for index, array in enumerate(inputs):
        inputs[index] = np.squeeze(array)
      inputs = torch.tensor(np.stack(feature_extractor(inputs)['pixel_values'], axis=0))
    
      # Send to appropriate computing device
      inputs = inputs.to(device)
      target = target.to(device)
     
      # Generate prediction
      prediction, loss = model(inputs, target)
        
      # Predicted class value using argmax
      predicted_class = np.argmax(prediction.cpu())
      value_predicted = list(valid_ds.class_to_idx.keys())[list(valid_ds.class_to_idx.values()).index(predicted_class)]
      value_target = list(valid_ds.class_to_idx.keys())[list(valid_ds.class_to_idx.values()).index(target)]
            
      # Show result
      plt.imshow(originalInput)
      plt.xlim(4096,0)
      plt.ylim(224,0)
      plt.title(f'Prediction: {value_predicted}')
      plt.show()
    
    return True

infer("C:/users/walte/downloads/final_model.pt", "C:/users/walte/desktop/faketwinkle.png")