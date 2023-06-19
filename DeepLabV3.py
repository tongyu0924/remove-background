import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
import matplotlib.pyplot as plt
import torchvision
import torchvision.models as models
from PIL import Image
import numpy as np

model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet101', pretrained=True)
model.eval()

input_image = Image.open('kouki.jpg')
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

segmentation_mask = output_predictions.detach().cpu().numpy()

alpha_channel = np.zeros_like(segmentation_mask, dtype=np.uint8)
alpha_channel[segmentation_mask != 0] = 255

input_image_np = np.array(input_image)
output_image_pil = Image.fromarray(input_image_np).convert("RGBA")
alpha_channel = np.zeros_like(segmentation_mask, dtype=np.uint8)
alpha_channel[segmentation_mask != 0] = 255

output_image_pil.putalpha(Image.fromarray(alpha_channel))

plt.imshow(output_image_pil)
plt.axis('off')
plt.show()

output_image_pil.save('output.png')
