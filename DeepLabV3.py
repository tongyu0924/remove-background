import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet101', pretrained=True)
model.eval()

input_image = Image.open("C:/Users/user/Downloads/kouki.jpg")
input_image = input_image.convert("RGB")
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0)

with torch.no_grad():
    output = model(input_batch)['out'][0]
output_predictions = output.argmax(0)

output_predictions_np = output_predictions.numpy().astype(np.float32)  # 將 PyTorch 張量轉換為 NumPy 數組
output_predictions_np = cv2.resize(output_predictions_np, (input_image.width, input_image.height))  # 調整預測結果的尺寸

output_predictions_np = (output_predictions_np * 255).astype(np.uint8)

input_np = np.array(input_image)
output_segmented = cv2.bitwise_and(input_np, input_np, mask=output_predictions_np)

plt.imshow(output_segmented)
plt.axis('off')
plt.show()
