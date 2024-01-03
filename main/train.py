import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torchvision.models as models
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

train_dir = '../dataset/train'
test_dir = '../dataset/test'


train_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
])


train_data = ImageFolder(train_dir, transform=train_transform)
test_data = ImageFolder(test_dir, transform=test_transform)

train_loader = DataLoader(train_data, batch_size=1024, shuffle=True, pin_memory=False)
test_loader = DataLoader(test_data, batch_size=1024, shuffle=True, pin_memory=False)

def train_model(model, criterion, optimizer, num_epochs=3):
    losses = {'train': [],'test': []}
    accuracies = {'train': [],'test': []}
    max_accuracy = 0
    val_preds = []
    val_labels = []
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))

        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()
           

                running_loss = 0.0
                running_corrects = 0
                for inputs, labels in tqdm(train_loader):
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    #loss = V(loss, requires_grad = True)

                    optimizer.zero_grad()
                    loss.backward()

                    optimizer.step()

                    preds_proba = nn.functional.softmax(outputs, 1)
                    preds = torch.argmax(preds_proba, 1)
                    running_loss += loss.item()
                    running_corrects += torch.sum(preds == labels.data)
                   

                epoch_loss = running_loss / len(
                    [phase])
                epoch_acc = running_corrects.double() / len(train_data)

                print('{} loss: {:.4f}, acc: {:.4f}'.format(phase,
                                                            epoch_loss,
                                                            epoch_acc))
                losses[phase].append(epoch_loss)
                accuracies[phase].append(epoch_acc.item())

            else:
                model.eval()
                with torch.no_grad():
                    running_loss = 0.0
                    running_corrects = 0
                    epoch_preds = []
                    epoch_labels = []

                    for inputs, labels in test_loader:
                        inputs = inputs.to(device)
                        labels = labels.to(device)

                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        #loss = V(loss, requires_grad = True)

                        preds_proba = nn.functional.softmax(outputs, 1)
                        preds = torch.argmax(preds_proba, 1)
                        running_loss += loss.item()
                        running_corrects += torch.sum(preds == labels.data)
                        epoch_preds+=preds.cpu().numpy().tolist()
                        epoch_labels+=labels.cpu().numpy().tolist()

                    epoch_loss = running_loss / len(test_data)
                    epoch_acc = running_corrects.double() / len(test_data)
                    
                    if epoch % 5 == 0:
                        print('{} loss: {:.4f}, acc: {:.4f}'.format(phase,
                                                                epoch_loss,
                                                                epoch_acc))
                    losses[phase].append(epoch_loss)
                    accuracies[phase].append(epoch_acc.item())
                    if max_accuracy < epoch_acc:
                        max_accuracy = epoch_acc
                        val_preds = epoch_preds
                        val_labels = epoch_labels

            
    return {"model": model, 
            "losses": losses, 
            "accuracies": accuracies, 
            "val_preds": epoch_preds, 
            "val_labels": epoch_labels}

    
def analyaze_model(model):
    model.eval()
    # Make predictions on the test data
    with torch.no_grad():
        predictions = []
        targets = []
        for batch in test_loader:
            images, labels = batch
            output = model(images.to(device))
            predictions.append(output.to('cpu').numpy())
            targets.append(labels.to('cpu').numpy())
        predictions = np.concatenate(predictions)
        targets = np.concatenate(targets)
    cm = confusion_matrix(targets, predictions.argmax(axis=1))
    report = classification_report(targets, predictions.argmax(axis=1))
    # Plot a confusion matrix
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(len(train_data.classes))
    plt.xticks(tick_marks, train_data.classes, rotation=45)
    plt.yticks(tick_marks, train_data.classes)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.show()
    print(report)
    accuracy = (predictions.argmax(axis=1) == targets).mean()
    print('Accuracy:', accuracy)
    
    
mobilenet_model = models.mobilenet_v3_small(pretrained=True).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, mobilenet_model.parameters()), lr=0.01)

mobilenet_model.fc2 = nn.Sequential(
               nn.ReLU(inplace=True),
               nn.Linear(1000, 128),
               nn.ReLU(inplace=True),
               nn.Linear(128, 7)).to(device)
for param in mobilenet_model.fc2.parameters():
    param.requires_grad = True 
    
trainin_output = train_model(mobilenet_model, criterion, optimizer, num_epochs=250)


torch.save(mobilenet_model, "../main/result/mobilenet.pt")

mobilenet_model = torch.load("../main/result/mobilenet.pt")
mobilenet_model.to(device)

analyaze_model(mobilenet_model)