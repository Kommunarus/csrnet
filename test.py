import json

import numpy as np
from matplotlib import pyplot as plt

from model import CSRNet
import torch
from torchvision import transforms
import dataset

transform=transforms.Compose([
                       transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                   ])

model = CSRNet()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
checkpoint = torch.load('01_model_best.pth.tar')
# checkpoint = torch.load('01_checkpoint.pth.tar')
model.load_state_dict(checkpoint['state_dict'])

with open('./test_list.json', 'r') as outfile:
    val_list = json.load(outfile)

test_loader = torch.utils.data.DataLoader(dataset.listDataset(val_list, shuffle=False, transform=transform, train=False),
    batch_size=1)

model.eval()


mae_list = []

for i, (img, target) in enumerate(test_loader):
    img = img.to(device)
    output = model(img)

    plt.figure(figsize=(12, 4))

    # Исходное изображение
    plt.subplot(1, 3, 1)
    img_cpu = img.cpu().squeeze(0).permute(1, 2, 0).numpy()
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_display = std * img_cpu + mean
    img_display = np.clip(img_display, 0, 1)
    plt.imshow(img_display)
    plt.title('Image')
    plt.axis('off')

    max_abs = 1e-5
    # Целевая карта плотности
    plt.subplot(1, 3, 2)
    target_cpu = target.cpu().squeeze(0).squeeze(0).numpy()
    plt.imshow(target_cpu, cmap='jet', )
    plt.title(f'Target (sum: {target_cpu.sum():.0f})')
    plt.axis('off')
    plt.colorbar()

    # Выход сети
    plt.subplot(1, 3, 3)
    output_cpu = output.detach().cpu().squeeze(0).squeeze(0).numpy()
    plt.imshow(output_cpu, cmap='jet', )

    plt.title(f'Output (sum: {output_cpu.sum():.0f})')
    plt.axis('off')
    plt.colorbar()

    plt.tight_layout()
    # Сохраняем вместо отображения
    plt.savefig(f'./pic_test/test_{i}.png', dpi=150, bbox_inches='tight')
    plt.close()