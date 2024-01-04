import torch
from torch import nn
from torchvision import transforms
from PIL import Image
import torchvision.models as models

device = torch.device('cpu')
print('Using device:', device)

test_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3,1,1)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
])



# image_path = '../sample/angry.jpg'
# image_path = '../sample/disgust.jpg'
# image_path = '../sample/fear.jpg'
# image_path = '../sample/happy.jpg'
# image_path = '../sample/neutral.jpg'
# image_path = '../sample/sad.jpg'
image_path = '../sample/surprise.jpg'
input_image = Image.open(image_path)
input_image = test_transform(input_image)
input_image = input_image.unsqueeze(0)
input_image = input_image.to(device)

    
mobilenet_model = models.mobilenet_v3_small(pretrained=True).to(device)

mobilenet_model.fc2 = nn.Sequential(
               nn.ReLU(inplace=True),
               nn.Linear(1000, 128),
               nn.ReLU(inplace=True),
               nn.Linear(128, 7)).to(device)

mobilenet_model = torch.load("../main/result/mobilenet.pt" ,map_location=device)
mobilenet_model.to(device)
mobilenet_model.eval()
with torch.no_grad():
    output = mobilenet_model(input_image)
    output = output.to('cpu').numpy()
    print(output.argmax(axis=1)[0])

'''
0 -> angry
1 -> disgust
2 -> fear
3 -> happy
4 -> neutral
5 -> sad
6 -> surprise
'''