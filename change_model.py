import torch
from torch import nn
from torchvision import transforms
from PIL import Image
import torchvision.models as models

device = torch.device('cpu')
print('Using device:', device)

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# image_path = '../sample/angry.jpg'
# image_path = '../sample/disgust.jpg'
# image_path = '../sample/fear.jpg'
# image_path = '../sample/happy.jpg'
# image_path = '../sample/neutral.jpg'
# image_path = '../sample/sad.jpg'
image_path = '../sample/fear.jpg'
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

mobilenet_model = torch.load("../main/result/mobilenet.pt", map_location=device)
mobilenet_model.to(device)
mobilenet_model.eval()
# with torch.no_grad():
#     output = mobilenet_model(input_image)
#     output = output.to('cpu').numpy()
#     print(output.argmax(axis=1)[0])

'''
0 -> angry
1 -> disgust
2 -> fear
3 -> happy
4 -> neutral
5 -> sad
6 -> surprise
'''
# 导出模型
# onnx_file_name = "emo_det.onnx"
# torch.onnx.export(mobilenet_model, input_image, onnx_file_name, verbose=True, input_names=['input'], output_names=['output'])
# 检测模型是否可用
onnx_model='./emo_det.onnx'
import onnx
# 我们可以使用异常处理的方法进行检验
try:
    # 当我们的模型不可用时，将会报出异常
    onnx.checker.check_model(onnx_model)
except onnx.checker.ValidationError as e:
    print("The model is invalid: %s"%e)
else:
    # 模型可用时，将不会报出异常，并会输出“The model is valid!”
    print("The model is valid!")
## 导入onnxruntime

import onnxruntime
# onnxruntime.InferenceSession用于获取一个 ONNX Runtime 推理器
ort_session = onnxruntime.InferenceSession(onnx_model)
# 构建字典的输入数据，字典的key需要与我们构建onnx模型时的input_names相同
# 输入的input_img 也需要改变为ndarray格式
ort_inputs = {'input': input_image.numpy()}
# 我们更建议使用下面这种方法,因为避免了手动输入key
# ort_inputs = {ort_session.get_inputs()[0].name:input_img}

# run是进行模型的推理，第一个参数为输出张量名的列表，一般情况可以设置为None
# 第二个参数为构建的输入值的字典
# 由于返回的结果被列表嵌套，因此我们需要进行[0]的索引
ort_output = ort_session.run(None,ort_inputs)[0]
print(ort_output.argmax(axis=1)[0])