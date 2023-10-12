from torchvision import models

alexNetCPU = {
    # https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html
    'model': models.alexnet(weights=models.AlexNet_Weights.DEFAULT),
    'weights': models.AlexNet_Weights.DEFAULT,
    'preprocess': models.AlexNet_Weights.DEFAULT.transforms(),
    'output_layer': 'classifier',
    'in_features': lambda model: model.classifier[1].in_features
}

alexNetGPU = {
    # https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html
    'model': models.alexnet(weights=models.AlexNet_Weights.DEFAULT),
    'weights': models.AlexNet_Weights.DEFAULT,
    'preprocess': models.AlexNet_Weights.DEFAULT.transforms(),
    'output_layer': 'classifier',
    'in_features': lambda model: model.classifier[1].in_features
}

resNet50CPU = {
    # https://arxiv.org/abs/1512.03385
    'model': models.resnet50(weights=models.ResNet50_Weights.DEFAULT),
    'weights': models.ResNet50_Weights.DEFAULT,
    'preprocess': models.ResNet50_Weights.DEFAULT.transforms(),
    'output_layer': 'classifier',
    'in_features': lambda model: model.classifier[0].in_features
}

resNet50GPU = {
    # https://arxiv.org/abs/1512.03385
    'model': models.resnet50(weights=models.ResNet50_Weights.DEFAULT),
    'weights': models.ResNet50_Weights.DEFAULT,
    'preprocess': models.ResNet50_Weights.DEFAULT.transforms(),
    'output_layer': 'classifier',
    'in_features': lambda model: model.classifier[0].in_features
}

denseNetCPU = {
    # https://arxiv.org/abs/1608.06993
    'model': models.densenet201(weights=models.DenseNet201_Weights.DEFAULT),
    'weights': models.DenseNet201_Weights.DEFAULT,
    'preprocess': models.DenseNet201_Weights.DEFAULT.transforms(),
    'output_layer': 'classifier',
    'in_features': lambda model: model.classifier[0].in_features
}

denseNetGPU = {
    # https://arxiv.org/abs/1608.06993
    'model': models.densenet201(weights=models.DenseNet201_Weights.DEFAULT),
    'weights': models.DenseNet201_Weights.DEFAULT,
    'preprocess': models.DenseNet201_Weights.DEFAULT.transforms(),
    'output_layer': 'classifier',
    'in_features': lambda model: model.classifier[0].in_features
}

mobileNetV3_smallCPU = {
    # https://arxiv.org/abs/1905.02244
    'model': models.mobilenet_v3_small(weights=models.mobilenetv3.MobileNet_V3_Small_Weights.DEFAULT),
    'weights': models.mobilenetv3.MobileNet_V3_Small_Weights.DEFAULT,
    'preprocess': models.mobilenetv3.MobileNet_V3_Small_Weights.DEFAULT.transforms(),
    'output_layer': 'classifier',
    'in_features': lambda model: model.classifier[0].in_features
}

mobileNetV3_smallGPU = {
    # https://arxiv.org/abs/1905.02244
    'model': models.mobilenet_v3_small(weights=models.mobilenetv3.MobileNet_V3_Small_Weights.DEFAULT),
    'weights': models.mobilenetv3.MobileNet_V3_Small_Weights.DEFAULT,
    'preprocess': models.mobilenetv3.MobileNet_V3_Small_Weights.DEFAULT.transforms(),
    'output_layer': 'classifier',
    'in_features': lambda model: model.classifier[0].in_features
}

swinTCPU = {
    # https://arxiv.org/abs/2103.14030
    'model': models.swin_t(weights=models.Swin_T_Weights.IMAGENET1K_V1.DEFAULT),
    'weights': models.Swin_T_Weights.IMAGENET1K_V1.DEFAULT,
    'preprocess': models.Swin_T_Weights.IMAGENET1K_V1.DEFAULT.transforms(),
    'output_layer': 'head',
    'in_features': lambda model: model.head.in_features
}

swinTGPU = {
    # https://arxiv.org/abs/2103.14030
    'model': models.swin_t(weights=models.Swin_T_Weights.IMAGENET1K_V1.DEFAULT),
    'weights': models.Swin_T_Weights.IMAGENET1K_V1.DEFAULT,
    'preprocess': models.Swin_T_Weights.IMAGENET1K_V1.DEFAULT.transforms(),
    'output_layer': 'head',
    'in_features': lambda model: model.head.in_features
}

"""# preprocess"""

import torch

gpu = 'has' if torch.cuda.is_available() else 'has no'
print(f'Current environment {gpu} GPU support')
deviceGPU = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
deviceCPU = torch.device("cpu")

!lscpu

print(f'GPU count = {torch.cuda.device_count() if torch.cuda.is_available() else "0"}')

for modelCPU in [resNet50CPU, alexNetCPU, denseNetCPU, mobileNetV3_smallCPU, swinTCPU]:
    modelCPU['model'].to(deviceCPU)
    modelCPU['preprocess'].to(deviceCPU)
    modelCPU['model'].eval()

for modelGPU in [resNet50GPU, alexNetGPU, denseNetGPU, mobileNetV3_smallGPU, swinTGPU]:
    modelGPU['model'].to(deviceGPU)
    modelGPU['preprocess'].to(deviceGPU)
    modelGPU['model'].eval()

"""# main"""

import os
import time
import copy
import math

import typing
from typing import Callable
from functools import partial

import numpy
import pandas
from pandas.core.arrays import boolean
import matplotlib.pyplot as plt
from requests import get

from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToPILImage, Compose, Resize, CenterCrop
from torchvision.io import read_image

strip_chars = ' \t'
tmp_file_name = 'tmp_file_name_for_image_download'

to_image = ToPILImage()

def classify(dataset: Dataset,
                preprocess: typing.Callable[[Tensor],Tensor],
                num_per_row: int, single_size: float,
                labels: typing.List[str],
                model_labels: typing.List[str] = None,
                model: typing.Callable[[Tensor], Tensor] = None,
                debug: typing.Any = False,
                num_of_classes: int = 1,
                vspace: float = 0.3
             ) -> None:

    detect_model = 0
    time_detect = 0

    num = len(dataset)
    fig, axs = plt.subplots(math.ceil(num/num_per_row), num_per_row, figsize=(
        single_size*num_per_row, (single_size + vspace)*(math.ceil(num/num_per_row))),
        sharex=True, sharey=True)
    for i in range(0, len(dataset)):
        try:
            image, label = dataset[i]
            pred = None

            if model is not None and model_labels is not None:
                start_time = time.perf_counter_ns()
                score = model(image.unsqueeze(0)).detach().squeeze(0).softmax(0)
                pred_index = numpy.flip(score.detach().cpu().argsort().numpy())[0]
                end_time = time.perf_counter_ns()
                pred = f'Detector: {(end_time - start_time) / 1_000_000:.0f}ms\n{model_labels[pred_index]}[{score[pred_index].item()*100:.0f}%]'
                #
                time_detect = ((end_time - start_time) / 1000000) + time_detect
                if model_labels[pred_index] == 'espresso maker' or model_labels[pred_index] == 'power drill':
                  detect_model = detect_model + 1
                #

            loc_fig = axs[i//num_per_row, i % num_per_row]
            loc_fig.imshow(to_image(preprocess(image)))
            title =f'\nActual: {labels[label]}[{label}]\n{pred}'
            loc_fig.title.set_text(title)
        except Exception as ex:
            if debug:
                raise ex
            print(f'Image {i} is failed to load: {str(ex)}')
    #
    print('Detector: ' + str(detect_model) + ' out of 100 models')
    print('Time: ' + str(time_detect/100))
    return detect_model, time_detect/100,
    #
    fig.subplots_adjust(wspace=0.3)
    plt.show()

def denormalize(dataset: Dataset, trans: typing.Any) -> Callable[[Tensor], Tensor]:
    image, label = dataset[0]
    std = torch.as_tensor(trans.std, dtype=image.dtype, device=image.device).view(-1, 1, 1)
    mean = torch.as_tensor(trans.mean, dtype=image.dtype, device=image.device).view(-1, 1, 1)
    return lambda img: img*std + mean


class UrlDataset(Dataset):

    def __init__(self, file: str, to_device, transform = None) -> None:
        self.file = file
        self.transform = transform
        self.dataset = pandas.read_csv(file, sep=';')
        self.classes = self.dataset['label'].unique()
        self.classes.sort()
        self.class_to_index = {self.classes[i] : i for i in range(len(self.classes))}
        self.device = to_device

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> typing.Tuple[torch.Tensor, str]:
        url = self.dataset.iloc[index]['url'].strip(strip_chars)
        with open(tmp_file_name, 'wb') as file:
            file.write(get(url).content)
        image = read_image(tmp_file_name).to(self.device)
        label = self.class_to_index[self.dataset.iloc[index]['label']]
        if self.transform:
            return self.transform(image), label
        return image, label


def train_model(model, dataloader: DataLoader, device: torch.device,
                critery, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 0.0
    best_epoch = -1

    process = {'train': {'loss': [], 'accuracy': []}, 'validate': {'loss': [], 'accuracy': []}}

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)
        epoch_loss = 0.0
        epoch_acc = 0.0
        for item in dataloader:
            if item['train'] == True:
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0
            dataset_size = 0
            for inputs, labels in item['loader']:
                dataset_size = dataset_size + 1
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(item['train'] == True):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = critery(outputs, labels)
                    if item['train'] == True:
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if item['train'] == True:
                scheduler.step()

            epoch_loss = running_loss / dataset_size
            epoch_acc = running_corrects.detach().cpu().double() / dataset_size

            if item['train'] == True:
                ptype = 'train'
            else:
                ptype = 'validate'

            process[ptype]['loss'].append(epoch_loss)
            process[ptype]['accuracy'].append(epoch_acc)

            print(f'[{epoch}][train={item["train"]}] Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if item['train'] == True and 1/epoch_loss > best_loss:
                best_loss = 1/epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                best_epoch = epoch

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val loss: {1/best_loss:4f} at epoch {best_epoch}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, process

"""# simple file"""

# Commented out IPython magic to ensure Python compatibility.
# %%writefile simple.csv
# label;url
# 
# микроскоп;https://github.com/Shaman-dp/neuroDataset/blob/main/102-6155-20532-detskiy-mikroskop-100x-1200x.jpg?raw=true
# микроскоп;https://github.com/Shaman-dp/neuroDataset/blob/main/1195857_v01_b.jpg?raw=true
# микроскоп;https://github.com/Shaman-dp/neuroDataset/blob/main/11971.600.jpg?raw=true
# микроскоп;https://github.com/Shaman-dp/neuroDataset/blob/main/148875096.jpg?raw=true
# микроскоп;https://github.com/Shaman-dp/neuroDataset/blob/main/1595273_v01_b.jpg?raw=true
# микроскоп;https://github.com/Shaman-dp/neuroDataset/blob/main/1_big.jpg?raw=true
# микроскоп;https://github.com/Shaman-dp/neuroDataset/blob/main/23337.600.jpg?raw=true
# микроскоп;https://github.com/Shaman-dp/neuroDataset/blob/main/28046.970.jpg?raw=true
# микроскоп;https://github.com/Shaman-dp/neuroDataset/blob/main/3.jpg?raw=true
# микроскоп;https://github.com/Shaman-dp/neuroDataset/blob/main/3870.970.jpg?raw=true
# микроскоп;https://github.com/Shaman-dp/neuroDataset/blob/main/41c4a7f010.jpg?raw=true
# микроскоп;https://github.com/Shaman-dp/neuroDataset/blob/main/45610044.jpg?raw=true
# микроскоп;https://github.com/Shaman-dp/neuroDataset/blob/main/4bc158b1f08b03162be756c495e7900c.jpg?raw=true
# микроскоп;https://github.com/Shaman-dp/neuroDataset/blob/main/4d2d3456c037c165f82b823320dc3338.jpg?raw=true
# микроскоп;https://github.com/Shaman-dp/neuroDataset/blob/main/6049003855.jpg?raw=true
# микроскоп;https://github.com/Shaman-dp/neuroDataset/blob/main/68ba85225ca2946d610cce170e3655fb.jpg?raw=true
# микроскоп;https://github.com/Shaman-dp/neuroDataset/blob/main/69051_levenhuk-rainbow-50l-plus-moonstone_00.jpg?raw=true
# микроскоп;https://github.com/Shaman-dp/neuroDataset/blob/main/73985_levenhuk-trinocular-microscope-med-10t_00.jpg?raw=true
# микроскоп;https://github.com/Shaman-dp/neuroDataset/blob/main/74004_levenhuk-med-40b-binocular-microscope_01.jpg?raw=true
# микроскоп;https://github.com/Shaman-dp/neuroDataset/blob/main/74008_levenhuk-med-45b-binocular-microscope_02.jpg?raw=true
# микроскоп;https://github.com/Shaman-dp/neuroDataset/blob/main/74319_bresser-mikroskop-junior-biolux-sel-40-1600x-zelenyj_00.jpg?raw=true
# микроскоп;https://github.com/Shaman-dp/neuroDataset/blob/main/74319_bresser-mikroskop-junior-biolux-sel-40-1600x-zelenyj_09.jpg?raw=true
# микроскоп;https://github.com/Shaman-dp/neuroDataset/blob/main/74322_bresser-mikroskop-junior-biolux-sel-40-1600x-sinij_09.jpg?raw=true
# микроскоп;https://github.com/Shaman-dp/neuroDataset/blob/main/75419_levenhuk-microscope-400m_00.jpg?raw=true
# микроскоп;https://github.com/Shaman-dp/neuroDataset/blob/main/75425_levenhuk-microscope-500b_00.jpg?raw=true
# микроскоп;https://github.com/Shaman-dp/neuroDataset/blob/main/75435_levenhuk-d400t-3-1m-digital-trinocular-microscope_00.jpg?raw=true
# микроскоп;https://github.com/Shaman-dp/neuroDataset/blob/main/7858388fc9e608c70424b5dd1567283f.jpg?raw=true
# микроскоп;https://github.com/Shaman-dp/neuroDataset/blob/main/8d709d7256a9e54454dc59065a33b39b3ad821ca14d1dc89924e030a33ca8c80.jpg?raw=true
# микроскоп;https://github.com/Shaman-dp/neuroDataset/blob/main/Micromet3m.jpg?raw=true
# микроскоп;https://github.com/Shaman-dp/neuroDataset/blob/main/ad95bd0455da1afcac911f74c37cf19d0bdb889e0b278e067fee385d97ccd775.jpg?raw=true
# микроскоп;https://github.com/Shaman-dp/neuroDataset/blob/main/binokmikroskop1-700x700.jpg?raw=true
# микроскоп;https://github.com/Shaman-dp/neuroDataset/blob/main/binokulyarnyj-mikroskop-n-300m.jpg?raw=true
# микроскоп;https://github.com/Shaman-dp/neuroDataset/blob/main/bnucmchh9zw52o1ktlftmmabmq3uh6zd.jpg?raw=true
# микроскоп;https://github.com/Shaman-dp/neuroDataset/blob/main/bresser_junior_8855000.jpg?raw=true
# микроскоп;https://github.com/Shaman-dp/neuroDataset/blob/main/df1c5bea2624439ccf79b5f948f68d85.jpg?raw=true
# микроскоп;https://github.com/Shaman-dp/neuroDataset/blob/main/editor5094.jpg?raw=true
# микроскоп;https://github.com/Shaman-dp/neuroDataset/blob/main/editor5898.jpg?raw=true
# микроскоп;https://github.com/Shaman-dp/neuroDataset/blob/main/editor8740.jpg?raw=true
# микроскоп;https://github.com/Shaman-dp/neuroDataset/blob/main/editor8742.jpg?raw=true
# микроскоп;https://github.com/Shaman-dp/neuroDataset/blob/main/ef960df9a09887d20c2bef899d9615b6.jpg?raw=true
# микроскоп;https://github.com/Shaman-dp/neuroDataset/blob/main/img_5.jpg?raw=true
# микроскоп;https://github.com/Shaman-dp/neuroDataset/blob/main/m_logo.jpg?raw=true
# микроскоп;https://github.com/Shaman-dp/neuroDataset/blob/main/micmed-5_1od_enl.jpg?raw=true
# микроскоп;https://github.com/Shaman-dp/neuroDataset/blob/main/microscope-levenhuk-720b.jpg?raw=true
# микроскоп;https://github.com/Shaman-dp/neuroDataset/blob/main/mikroskop-levenhuk-labzz-m2.jpg?raw=true
# микроскоп;https://github.com/Shaman-dp/neuroDataset/blob/main/mikroskop.jpg?raw=true
# микроскоп;https://github.com/Shaman-dp/neuroDataset/blob/main/mikroskop_biologicheskiy_mikromed_s_13_1.jpg?raw=true
# микроскоп;https://github.com/Shaman-dp/neuroDataset/blob/main/mikroskop_discovery_nano_terra_s_knigoy_1637599206_1.jpg?raw=true
# микроскоп;https://github.com/Shaman-dp/neuroDataset/blob/main/mikroskopy_3260f41c541737d_800x600.jpg?raw=true
# микроскоп;https://github.com/Shaman-dp/neuroDataset/blob/main/shop_items_catalog_image37040.jpg?raw=true
# лобзик;https://github.com/Shaman-dp/neuroDataset/blob/main/image_1.jpg?raw=true
# лобзик;https://github.com/Shaman-dp/neuroDataset/blob/main/image_2.jpg?raw=true
# лобзик;https://github.com/Shaman-dp/neuroDataset/blob/main/image_3.jpg?raw=true
# лобзик;https://github.com/Shaman-dp/neuroDataset/blob/main/image_4.jpg?raw=true
# лобзик;https://github.com/Shaman-dp/neuroDataset/blob/main/image_5.jpg?raw=true
# лобзик;https://github.com/Shaman-dp/neuroDataset/blob/main/image_6.jpg?raw=true
# лобзик;https://github.com/Shaman-dp/neuroDataset/blob/main/image_7.jpg?raw=true
# лобзик;https://github.com/Shaman-dp/neuroDataset/blob/main/image_8.jpg?raw=true
# лобзик;https://github.com/Shaman-dp/neuroDataset/blob/main/image_9.jpg?raw=true
# лобзик;https://github.com/Shaman-dp/neuroDataset/blob/main/image_10.jpg?raw=true
# лобзик;https://github.com/Shaman-dp/neuroDataset/blob/main/image_11.jpg?raw=true
# лобзик;https://github.com/Shaman-dp/neuroDataset/blob/main/image_12.jpg?raw=true
# лобзик;https://github.com/Shaman-dp/neuroDataset/blob/main/image_13.jpg?raw=true
# лобзик;https://github.com/Shaman-dp/neuroDataset/blob/main/image_14.jpg?raw=true
# лобзик;https://github.com/Shaman-dp/neuroDataset/blob/main/image_15.jpg?raw=true
# лобзик;https://github.com/Shaman-dp/neuroDataset/blob/main/image_16.jpg?raw=true
# лобзик;https://github.com/Shaman-dp/neuroDataset/blob/main/image_17.jpg?raw=true
# лобзик;https://github.com/Shaman-dp/neuroDataset/blob/main/image_18.jpg?raw=true
# лобзик;https://github.com/Shaman-dp/neuroDataset/blob/main/image_19.jpg?raw=true
# лобзик;https://github.com/Shaman-dp/neuroDataset/blob/main/image_20.jpg?raw=true
# лобзик;https://github.com/Shaman-dp/neuroDataset/blob/main/image_21.jpg?raw=true
# лобзик;https://github.com/Shaman-dp/neuroDataset/blob/main/image_22.jpg?raw=true
# лобзик;https://github.com/Shaman-dp/neuroDataset/blob/main/image_23.jpg?raw=true
# лобзик;https://github.com/Shaman-dp/neuroDataset/blob/main/image_24.jpg?raw=true
# лобзик;https://github.com/Shaman-dp/neuroDataset/blob/main/image_25.jpg?raw=true
# лобзик;https://github.com/Shaman-dp/neuroDataset/blob/main/image_26.jpg?raw=true
# лобзик;https://github.com/Shaman-dp/neuroDataset/blob/main/image_27.jpg?raw=true
# лобзик;https://github.com/Shaman-dp/neuroDataset/blob/main/image_28.jpg?raw=true
# лобзик;https://github.com/Shaman-dp/neuroDataset/blob/main/image_29.jpg?raw=true
# лобзик;https://github.com/Shaman-dp/neuroDataset/blob/main/image_30.jpg?raw=true
# лобзик;https://github.com/Shaman-dp/neuroDataset/blob/main/image_31.jpg?raw=true
# лобзик;https://github.com/Shaman-dp/neuroDataset/blob/main/image_32.jpg?raw=true
# лобзик;https://github.com/Shaman-dp/neuroDataset/blob/main/image_33.jpg?raw=true
# лобзик;https://github.com/Shaman-dp/neuroDataset/blob/main/image_34.jpg?raw=true
# лобзик;https://github.com/Shaman-dp/neuroDataset/blob/main/image_35.jpg?raw=true
# лобзик;https://github.com/Shaman-dp/neuroDataset/blob/main/image_36.jpg?raw=true
# лобзик;https://github.com/Shaman-dp/neuroDataset/blob/main/image_37.jpg?raw=true
# лобзик;https://github.com/Shaman-dp/neuroDataset/blob/main/image_38.jpg?raw=true
# лобзик;https://github.com/Shaman-dp/neuroDataset/blob/main/image_39.jpg?raw=true
# лобзик;https://github.com/Shaman-dp/neuroDataset/blob/main/image_40.jpg?raw=true
# лобзик;https://github.com/Shaman-dp/neuroDataset/blob/main/image_41.jpg?raw=true
# лобзик;https://github.com/Shaman-dp/neuroDataset/blob/main/image_42.jpg?raw=true
# лобзик;https://github.com/Shaman-dp/neuroDataset/blob/main/image_43.jpg?raw=true
# лобзик;https://github.com/Shaman-dp/neuroDataset/blob/main/image_44.jpg?raw=true
# лобзик;https://github.com/Shaman-dp/neuroDataset/blob/main/image_45.jpg?raw=true
# лобзик;https://github.com/Shaman-dp/neuroDataset/blob/main/image_46.jpg?raw=true
# лобзик;https://github.com/Shaman-dp/neuroDataset/blob/main/image_47.jpg?raw=true
# лобзик;https://github.com/Shaman-dp/neuroDataset/blob/main/image_48.jpg?raw=true
# лобзик;https://github.com/Shaman-dp/neuroDataset/blob/main/image_49.jpg?raw=true
# лобзик;https://github.com/Shaman-dp/neuroDataset/blob/main/image_50.jpg?raw=true

"""# alexNet"""

model_alexNet = alexNetCPU
transform = model_alexNet['preprocess']
simple = UrlDataset("simple.csv", deviceCPU, transform)

print('-'*70)
print(f'Model on CPU: { model_alexNet["model"].__class__.__name__}')
print(f'Number of parameters: {sum(item.numel() for item in model_alexNet["model"].parameters())}')

num_per_row = 5
single_size = 3.5
vspace = 0.4

model_alexNet_detect_CPU, model_alexNet_detect_time_CPU = classify(simple, denormalize(simple, transform), num_per_row=num_per_row, single_size=single_size, vspace = vspace,
            labels = simple.classes, model=model_alexNet['model'], model_labels=model_alexNet['weights'].meta["categories"],
            debug=True)

model_alexNet = alexNetGPU
transform = model_alexNet['preprocess']
simple = UrlDataset("simple.csv", deviceGPU, transform)

print('-'*70)
print(f'Model on GPU: { model_alexNet["model"].__class__.__name__}')
print(f'Number of parameters: {sum(item.numel() for item in model_alexNet["model"].parameters())}')

num_per_row = 5
single_size = 3.5
vspace = 0.4

model_alexNet_detect_GPU, model_alexNet_detect_time_GPU = classify(simple, denormalize(simple, transform), num_per_row=num_per_row, single_size=single_size, vspace = vspace,
            labels = simple.classes, model=model_alexNet['model'], model_labels=model_alexNet['weights'].meta["categories"],
            debug=True)

"""# resNet50"""

model_resNet50 = resNet50CPU
transform = model_resNet50['preprocess']
simple = UrlDataset("simple.csv", deviceCPU, transform)

print('-'*70)
print(f'Model on CPU: { model_resNet50["model"].__class__.__name__}')
print(f'Number of parameters: {sum(item.numel() for item in model_resNet50["model"].parameters())}')

num_per_row = 5
single_size = 3.5
vspace = 0.4

model_resNet50_detect_CPU, model_resNet50_detect_time_CPU = classify(simple, denormalize(simple, transform), num_per_row=num_per_row, single_size=single_size, vspace = vspace,
            labels = simple.classes, model=model_resNet50['model'], model_labels=model_resNet50['weights'].meta["categories"],
            debug=True)

model_resNet50 = resNet50GPU
transform = model_resNet50['preprocess']
simple = UrlDataset("simple.csv", deviceGPU, transform)

print('-'*70)
print(f'Model on GPU: { model_resNet50["model"].__class__.__name__}')
print(f'Number of parameters: {sum(item.numel() for item in model_resNet50["model"].parameters())}')

num_per_row = 5
single_size = 3.5
vspace = 0.4

model_resNet50_detect_GPU, model_resNet50_detect_time_GPU = classify(simple, denormalize(simple, transform), num_per_row=num_per_row, single_size=single_size, vspace = vspace,
            labels = simple.classes, model=model_resNet50['model'], model_labels=model_resNet50['weights'].meta["categories"],
            debug=True)

"""# denseNet"""

model_denseNet = denseNetCPU
transform = model_denseNet['preprocess']
simple = UrlDataset("simple.csv", deviceCPU, transform)

print('-'*70)
print(f'Model on CPU: { model_denseNet["model"].__class__.__name__}')
print(f'Number of parameters: {sum(item.numel() for item in model_denseNet["model"].parameters())}')

num_per_row = 5
single_size = 3.5
vspace = 0.4

model_denseNet_detect_CPU, model_denseNet_detect_time_CPU = classify(simple, denormalize(simple, transform), num_per_row=num_per_row, single_size=single_size, vspace = vspace,
            labels = simple.classes, model=model_denseNet['model'], model_labels=model_denseNet['weights'].meta["categories"],
            debug=True)

model_denseNet = denseNetGPU
transform = model_denseNet['preprocess']
simple = UrlDataset("simple.csv", deviceGPU, transform)

print('-'*70)
print(f'Model on GPU: { model_denseNet["model"].__class__.__name__}')
print(f'Number of parameters: {sum(item.numel() for item in model_denseNet["model"].parameters())}')

num_per_row = 5
single_size = 3.5
vspace = 0.4

model_denseNet_detect_GPU, model_denseNet_detect_time_GPU = classify(simple, denormalize(simple, transform), num_per_row=num_per_row, single_size=single_size, vspace = vspace,
            labels = simple.classes, model=model_denseNet['model'], model_labels=model_denseNet['weights'].meta["categories"],
            debug=True)

"""# mobileNetV3_small"""

model_mobileNetV3_small = mobileNetV3_smallCPU
transform = model_mobileNetV3_small['preprocess']
simple = UrlDataset("simple.csv", deviceCPU, transform)

print('-'*70)
print(f'Model on CPU: { model_mobileNetV3_small["model"].__class__.__name__}')
print(f'Number of parameters: {sum(item.numel() for item in model_mobileNetV3_small["model"].parameters())}')

num_per_row = 5
single_size = 3.5
vspace = 0.4

model_mobileNetV3_small_detect_CPU, model_mobileNetV3_small_detect_time_CPU = classify(simple, denormalize(simple, transform), num_per_row=num_per_row, single_size=single_size, vspace = vspace,
            labels = simple.classes, model=model_mobileNetV3_small['model'], model_labels=model_mobileNetV3_small['weights'].meta["categories"],
            debug=True)

model_mobileNetV3_small = mobileNetV3_smallGPU
transform = model_mobileNetV3_small['preprocess']
simple = UrlDataset("simple.csv", deviceGPU, transform)

print('-'*70)
print(f'Model on GPU: { model_mobileNetV3_small["model"].__class__.__name__}')
print(f'Number of parameters: {sum(item.numel() for item in model_mobileNetV3_small["model"].parameters())}')

num_per_row = 5
single_size = 3.5
vspace = 0.4

model_mobileNetV3_small_detect_GPU, model_mobileNetV3_small_detect_time_GPU = classify(simple, denormalize(simple, transform), num_per_row=num_per_row, single_size=single_size, vspace = vspace,
            labels = simple.classes, model=model_mobileNetV3_small['model'], model_labels=model_mobileNetV3_small['weights'].meta["categories"],
            debug=True)

"""# swinT"""

model_swinT = swinTGPU
transform = model_swinT['preprocess']
simple = UrlDataset("simple.csv", deviceGPU, transform)

print('-'*70)
print(f'Model on GPU: { model_swinT["model"].__class__.__name__}')
print(f'Number of parameters: {sum(item.numel() for item in model_swinT["model"].parameters())}')

num_per_row = 5
single_size = 3.5
vspace = 0.4

model_swinT_detect_GPU, model_swinT_detect_time_GPU = classify(simple, denormalize(simple, transform), num_per_row=num_per_row, single_size=single_size, vspace = vspace,
            labels = simple.classes, model=model_swinT['model'], model_labels=model_swinT['weights'].meta["categories"],
            debug=True)

model_swinT = swinTCPU
transform = model_swinT['preprocess']
simple = UrlDataset("simple.csv", deviceCPU, transform)

print('-'*70)
print(f'Model on CPU: { model_swinT["model"].__class__.__name__}')
print(f'Number of parameters: {sum(item.numel() for item in model_swinT["model"].parameters())}')

num_per_row = 5
single_size = 3.5
vspace = 0.4

model_swinT_detect_CPU, model_swinT_detect_time_CPU = classify(simple, denormalize(simple, transform), num_per_row=num_per_row, single_size=single_size, vspace = vspace,
            labels = simple.classes, model=model_swinT['model'], model_labels=model_swinT['weights'].meta["categories"],
            debug=True)

"""# сравнение"""

model_title = ['alexNet', 'resNet50', 'denseNet', 'mobileNetV3', 'swinT']
object_detect_CPU = [model_alexNet_detect_CPU,
                 model_resNet50_detect_CPU,
                 model_denseNet_detect_CPU,
                 model_mobileNetV3_small_detect_CPU,
                 model_swinT_detect_CPU]

# object_detect_GPU = [model_alexNet_detect_GPU,
#                  model_resNet50_detect_GPU,
#                  model_denseNet_detect_GPU,
#                  model_mobileNetV3_small_detect_GPU,
#                  model_swinT_detect_GPU]

width = 0.4
x = numpy.arange(5)
fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, object_detect_CPU, width, label = 'CPU')
# rects2 = ax.bar(x + width/2, object_detect_GPU, width, label = 'GPU')
ax.set_xticks(x)
ax.set_xticklabels(model_title)
ax.legend()

time_detect_CPU = [model_alexNet_detect_time_CPU,
                 model_resNet50_detect_time_CPU,
                 model_denseNet_detect_time_CPU,
                 model_mobileNetV3_small_detect_time_CPU,
                 model_swinT_detect_time_CPU]

time_detect_GPU = [model_alexNet_detect_time_GPU,
                 model_resNet50_detect_time_GPU,
                 model_denseNet_detect_time_GPU,
                 model_mobileNetV3_small_detect_time_GPU,
                 model_swinT_detect_time_GPU]

width = 0.4
x = numpy.arange(5)
fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, time_detect_CPU, width, label = 'CPU', color='black')
rects2 = ax.bar(x + width/2, time_detect_GPU, width, label = 'GPU', color='#A3A3A3')
ax.set_xticks(x)
ax.set_xticklabels(model_title)
ax.legend()

"""# train file"""

# Commented out IPython magic to ensure Python compatibility.
# %%writefile train.csv
# label;url
# 
# microscope;https://github.com/Shaman-dp/neuroDataset/blob/main/102-6155-20532-detskiy-mikroskop-100x-1200x.jpg?raw=true
# microscope;https://github.com/Shaman-dp/neuroDataset/blob/main/1195857_v01_b.jpg?raw=true
# microscope;https://github.com/Shaman-dp/neuroDataset/blob/main/11971.600.jpg?raw=true
# microscope;https://github.com/Shaman-dp/neuroDataset/blob/main/148875096.jpg?raw=true
# microscope;https://github.com/Shaman-dp/neuroDataset/blob/main/1595273_v01_b.jpg?raw=true
# microscope;https://github.com/Shaman-dp/neuroDataset/blob/main/1_big.jpg?raw=true
# microscope;https://github.com/Shaman-dp/neuroDataset/blob/main/23337.600.jpg?raw=true
# microscope;https://github.com/Shaman-dp/neuroDataset/blob/main/28046.970.jpg?raw=true
# microscope;https://github.com/Shaman-dp/neuroDataset/blob/main/3.jpg?raw=true
# microscope;https://github.com/Shaman-dp/neuroDataset/blob/main/3870.970.jpg?raw=true
# microscope;https://github.com/Shaman-dp/neuroDataset/blob/main/41c4a7f010.jpg?raw=true
# microscope;https://github.com/Shaman-dp/neuroDataset/blob/main/45610044.jpg?raw=true
# microscope;https://github.com/Shaman-dp/neuroDataset/blob/main/4bc158b1f08b03162be756c495e7900c.jpg?raw=true
# microscope;https://github.com/Shaman-dp/neuroDataset/blob/main/4d2d3456c037c165f82b823320dc3338.jpg?raw=true
# microscope;https://github.com/Shaman-dp/neuroDataset/blob/main/6049003855.jpg?raw=true
# microscope;https://github.com/Shaman-dp/neuroDataset/blob/main/68ba85225ca2946d610cce170e3655fb.jpg?raw=true
# microscope;https://github.com/Shaman-dp/neuroDataset/blob/main/69051_levenhuk-rainbow-50l-plus-moonstone_00.jpg?raw=true
# microscope;https://github.com/Shaman-dp/neuroDataset/blob/main/73985_levenhuk-trinocular-microscope-med-10t_00.jpg?raw=true
# microscope;https://github.com/Shaman-dp/neuroDataset/blob/main/74004_levenhuk-med-40b-binocular-microscope_01.jpg?raw=true
# microscope;https://github.com/Shaman-dp/neuroDataset/blob/main/74008_levenhuk-med-45b-binocular-microscope_02.jpg?raw=true
# microscope;https://github.com/Shaman-dp/neuroDataset/blob/main/74319_bresser-mikroskop-junior-biolux-sel-40-1600x-zelenyj_00.jpg?raw=true
# microscope;https://github.com/Shaman-dp/neuroDataset/blob/main/74319_bresser-mikroskop-junior-biolux-sel-40-1600x-zelenyj_09.jpg?raw=true
# microscope;https://github.com/Shaman-dp/neuroDataset/blob/main/74322_bresser-mikroskop-junior-biolux-sel-40-1600x-sinij_09.jpg?raw=true
# microscope;https://github.com/Shaman-dp/neuroDataset/blob/main/75419_levenhuk-microscope-400m_00.jpg?raw=true
# microscope;https://github.com/Shaman-dp/neuroDataset/blob/main/75425_levenhuk-microscope-500b_00.jpg?raw=true
# microscope;https://github.com/Shaman-dp/neuroDataset/blob/main/75435_levenhuk-d400t-3-1m-digital-trinocular-microscope_00.jpg?raw=true
# microscope;https://github.com/Shaman-dp/neuroDataset/blob/main/7858388fc9e608c70424b5dd1567283f.jpg?raw=true
# microscope;https://github.com/Shaman-dp/neuroDataset/blob/main/8d709d7256a9e54454dc59065a33b39b3ad821ca14d1dc89924e030a33ca8c80.jpg?raw=true
# microscope;https://github.com/Shaman-dp/neuroDataset/blob/main/Micromet3m.jpg?raw=true
# microscope;https://github.com/Shaman-dp/neuroDataset/blob/main/ad95bd0455da1afcac911f74c37cf19d0bdb889e0b278e067fee385d97ccd775.jpg?raw=true
# microscope;https://github.com/Shaman-dp/neuroDataset/blob/main/binokmikroskop1-700x700.jpg?raw=true
# microscope;https://github.com/Shaman-dp/neuroDataset/blob/main/binokulyarnyj-mikroskop-n-300m.jpg?raw=true
# microscope;https://github.com/Shaman-dp/neuroDataset/blob/main/bnucmchh9zw52o1ktlftmmabmq3uh6zd.jpg?raw=true
# microscope;https://github.com/Shaman-dp/neuroDataset/blob/main/bresser_junior_8855000.jpg?raw=true
# microscope;https://github.com/Shaman-dp/neuroDataset/blob/main/df1c5bea2624439ccf79b5f948f68d85.jpg?raw=true
# microscope;https://github.com/Shaman-dp/neuroDataset/blob/main/editor5094.jpg?raw=true
# microscope;https://github.com/Shaman-dp/neuroDataset/blob/main/editor5898.jpg?raw=true
# microscope;https://github.com/Shaman-dp/neuroDataset/blob/main/editor8740.jpg?raw=true
# microscope;https://github.com/Shaman-dp/neuroDataset/blob/main/editor8742.jpg?raw=true
# microscope;https://github.com/Shaman-dp/neuroDataset/blob/main/ef960df9a09887d20c2bef899d9615b6.jpg?raw=true
# jigsaw;https://github.com/Shaman-dp/neuroDataset/blob/main/image_1.jpg?raw=true
# jigsaw;https://github.com/Shaman-dp/neuroDataset/blob/main/image_2.jpg?raw=true
# jigsaw;https://github.com/Shaman-dp/neuroDataset/blob/main/image_3.jpg?raw=true
# jigsaw;https://github.com/Shaman-dp/neuroDataset/blob/main/image_4.jpg?raw=true
# jigsaw;https://github.com/Shaman-dp/neuroDataset/blob/main/image_5.jpg?raw=true
# jigsaw;https://github.com/Shaman-dp/neuroDataset/blob/main/image_6.jpg?raw=true
# jigsaw;https://github.com/Shaman-dp/neuroDataset/blob/main/image_7.jpg?raw=true
# jigsaw;https://github.com/Shaman-dp/neuroDataset/blob/main/image_8.jpg?raw=true
# jigsaw;https://github.com/Shaman-dp/neuroDataset/blob/main/image_9.jpg?raw=true
# jigsaw;https://github.com/Shaman-dp/neuroDataset/blob/main/image_10.jpg?raw=true
# jigsaw;https://github.com/Shaman-dp/neuroDataset/blob/main/image_11.jpg?raw=true
# jigsaw;https://github.com/Shaman-dp/neuroDataset/blob/main/image_12.jpg?raw=true
# jigsaw;https://github.com/Shaman-dp/neuroDataset/blob/main/image_13.jpg?raw=true
# jigsaw;https://github.com/Shaman-dp/neuroDataset/blob/main/image_14.jpg?raw=true
# jigsaw;https://github.com/Shaman-dp/neuroDataset/blob/main/image_15.jpg?raw=true
# jigsaw;https://github.com/Shaman-dp/neuroDataset/blob/main/image_16.jpg?raw=true
# jigsaw;https://github.com/Shaman-dp/neuroDataset/blob/main/image_17.jpg?raw=true
# jigsaw;https://github.com/Shaman-dp/neuroDataset/blob/main/image_18.jpg?raw=true
# jigsaw;https://github.com/Shaman-dp/neuroDataset/blob/main/image_19.jpg?raw=true
# jigsaw;https://github.com/Shaman-dp/neuroDataset/blob/main/image_20.jpg?raw=true
# jigsaw;https://github.com/Shaman-dp/neuroDataset/blob/main/image_21.jpg?raw=true
# jigsaw;https://github.com/Shaman-dp/neuroDataset/blob/main/image_22.jpg?raw=true
# jigsaw;https://github.com/Shaman-dp/neuroDataset/blob/main/image_23.jpg?raw=true
# jigsaw;https://github.com/Shaman-dp/neuroDataset/blob/main/image_24.jpg?raw=true
# jigsaw;https://github.com/Shaman-dp/neuroDataset/blob/main/image_25.jpg?raw=true
# jigsaw;https://github.com/Shaman-dp/neuroDataset/blob/main/image_26.jpg?raw=true
# jigsaw;https://github.com/Shaman-dp/neuroDataset/blob/main/image_27.jpg?raw=true
# jigsaw;https://github.com/Shaman-dp/neuroDataset/blob/main/image_28.jpg?raw=true
# jigsaw;https://github.com/Shaman-dp/neuroDataset/blob/main/image_29.jpg?raw=true
# jigsaw;https://github.com/Shaman-dp/neuroDataset/blob/main/image_30.jpg?raw=true
# jigsaw;https://github.com/Shaman-dp/neuroDataset/blob/main/image_31.jpg?raw=true
# jigsaw;https://github.com/Shaman-dp/neuroDataset/blob/main/image_32.jpg?raw=true
# jigsaw;https://github.com/Shaman-dp/neuroDataset/blob/main/image_33.jpg?raw=true
# jigsaw;https://github.com/Shaman-dp/neuroDataset/blob/main/image_34.jpg?raw=true
# jigsaw;https://github.com/Shaman-dp/neuroDataset/blob/main/image_35.jpg?raw=true
# jigsaw;https://github.com/Shaman-dp/neuroDataset/blob/main/image_36.jpg?raw=true
# jigsaw;https://github.com/Shaman-dp/neuroDataset/blob/main/image_37.jpg?raw=true
# jigsaw;https://github.com/Shaman-dp/neuroDataset/blob/main/image_38.jpg?raw=true
# jigsaw;https://github.com/Shaman-dp/neuroDataset/blob/main/image_39.jpg?raw=true
# jigsaw;https://github.com/Shaman-dp/neuroDataset/blob/main/image_40.jpg?raw=true

"""# test file"""

# Commented out IPython magic to ensure Python compatibility.
# %%writefile test.csv
# label;url
# 
# microscope;https://github.com/Shaman-dp/neuroDataset/blob/main/img_5.jpg?raw=true
# microscope;https://github.com/Shaman-dp/neuroDataset/blob/main/m_logo.jpg?raw=true
# microscope;https://github.com/Shaman-dp/neuroDataset/blob/main/micmed-5_1od_enl.jpg?raw=true
# microscope;https://github.com/Shaman-dp/neuroDataset/blob/main/microscope-levenhuk-720b.jpg?raw=true
# microscope;https://github.com/Shaman-dp/neuroDataset/blob/main/mikroskop-levenhuk-labzz-m2.jpg?raw=true
# microscope;https://github.com/Shaman-dp/neuroDataset/blob/main/mikroskop.jpg?raw=true
# microscope;https://github.com/Shaman-dp/neuroDataset/blob/main/mikroskop_biologicheskiy_mikromed_s_13_1.jpg?raw=true
# microscope;https://github.com/Shaman-dp/neuroDataset/blob/main/mikroskop_discovery_nano_terra_s_knigoy_1637599206_1.jpg?raw=true
# microscope;https://github.com/Shaman-dp/neuroDataset/blob/main/mikroskopy_3260f41c541737d_800x600.jpg?raw=true
# microscope;https://github.com/Shaman-dp/neuroDataset/blob/main/shop_items_catalog_image37040.jpg?raw=true
# jigsaw;https://github.com/Shaman-dp/neuroDataset/blob/main/image_41.jpg?raw=true
# jigsaw;https://github.com/Shaman-dp/neuroDataset/blob/main/image_42.jpg?raw=true
# jigsaw;https://github.com/Shaman-dp/neuroDataset/blob/main/image_43.jpg?raw=true
# jigsaw;https://github.com/Shaman-dp/neuroDataset/blob/main/image_44.jpg?raw=true
# jigsaw;https://github.com/Shaman-dp/neuroDataset/blob/main/image_45.jpg?raw=true
# jigsaw;https://github.com/Shaman-dp/neuroDataset/blob/main/image_46.jpg?raw=true
# jigsaw;https://github.com/Shaman-dp/neuroDataset/blob/main/image_47.jpg?raw=true
# jigsaw;https://github.com/Shaman-dp/neuroDataset/blob/main/image_48.jpg?raw=true
# jigsaw;https://github.com/Shaman-dp/neuroDataset/blob/main/image_49.jpg?raw=true
# jigsaw;https://github.com/Shaman-dp/neuroDataset/blob/main/image_50.jpg?raw=true

"""# Test"""

transform = model_swinT['preprocess']
train = UrlDataset("train.csv", deviceCPU, transform)
test = UrlDataset('test.csv', deviceCPU, transform)

num_per_row = 5
single_size = 3.5
vspace = 0.3

print('-'*70)
print(f'Model: { model_swinT["model"].__class__.__name__}')
print(f'Number of parameters: {sum(item.numel() for item in model_swinT["model"].parameters())}')

print('-'*30  + ' Train dataset ' + '-'*30)
classify(train, denormalize(train, transform), num_per_row=num_per_row, single_size=single_size, vspace=vspace,
            labels = train.classes, model=model_swinT['model'], model_labels=model_swinT['weights'].meta["categories"])
print('-'*30  + ' Test dataset ' + '-'*30)
classify(test, denormalize(test, transform), num_per_row=num_per_row, single_size=single_size, vspace=vspace,
            labels = test.classes, model=model_swinT['model'], model_labels=model_swinT['weights'].meta["categories"])

image, label = simple[0]
feature_extractor = copy.deepcopy(model_swinT['model'])
result = feature_extractor(image.unsqueeze(0)).cpu().detach().squeeze(0).numpy()
print(f'Размерность выходов исходной модели: {result.shape}')
setattr(feature_extractor, model_swinT['output_layer'], nn.Identity())
result = feature_extractor(image.unsqueeze(0)).cpu().detach().squeeze(0).numpy()
print(f'Размерность выходов модели после замены выходного классификатора на единичное преобразование: {result.shape}')

output = []
for image, label in torch.utils.data.ConcatDataset([train, test]):
    result = {'label': label, 'feature': feature_extractor(image.unsqueeze(0)).cpu().detach().squeeze(0).numpy()}
    output.append(result)
features = numpy.concatenate([numpy.expand_dims(record['feature'], 0) for record in output])
labels = [record['label'] for i, record in enumerate(output)]

totalU, totalS, totalV = torch.pca_lowrank(torch.from_numpy(features), q=2)

series = {}

for key in ['microscope', 'jigsaw']:
    points = numpy.concatenate([numpy.expand_dims(totalU[i,:].cpu().detach().numpy(), 0)
        for i, label in enumerate(labels) if train.classes[label] == key])
    series[key] = numpy.moveaxis(points, 1, 0)

plt.figure()
plt.plot(series['microscope'][0], series['microscope'][1], 'ro', series['jigsaw'][0], series['jigsaw'][1], 'bo')
plt.legend(['microscope', 'jigsaw'])
plt.show()

tunned_model = copy.deepcopy(model_swinT['model'])
for param in tunned_model.parameters():
    param.requires_grad = False

print(getattr(tunned_model, model_swinT['output_layer']))

in_features = model_swinT['in_features'](tunned_model)
print(in_features)

setattr(tunned_model, model_swinT['output_layer'], torch.nn.Sequential(torch.nn.Linear(in_features, 2), torch.nn.Softmax(dim=1)))

print(f"num of adjustable parameters = {sum(i.numel() for i in getattr(tunned_model, model_swinT['output_layer']).parameters())}")

dataloader = [
    {
        'train': True,
        'loader': DataLoader(train, batch_size=1, shuffle=True, num_workers=0)
    },
    {
        'train': False,
        'loader': DataLoader(test, batch_size=1, shuffle=True, num_workers=0)
    }
]

tunned_model.to(deviceCPU)
critery = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(getattr(tunned_model, model_swinT['output_layer']).parameters(), lr=0.001, momentum=0.9)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5,verbose=True)

model_conv, process = train_model(model=tunned_model, dataloader=dataloader, device=deviceCPU,
                         critery=critery, optimizer=optimizer, scheduler=lr_scheduler,
                         num_epochs=9)

fig, axs = plt.subplots(2,2, figsize=(16,10), sharex=True)
axs[0,0].plot(process['train']['loss'])
axs[0,0].title.set_text('Эволюции ошибки на выборке для обучения')
axs[0,1].plot(process['train']['accuracy'])
axs[0,1].title.set_text('Эволюции ошибки на выборке для обучения')
axs[1,0].plot(process['validate']['loss'])
axs[1,0].title.set_text('Эволюции ошибки на выборке для проверки')
axs[1,1].plot(process['validate']['accuracy'])
axs[1,1].title.set_text('Эволюции ошибки на выборке для проверки')
axs[1,0].set_xlabel('Итерация обучения')
axs[1,1].set_xlabel('Итерация обучения')
axs[0,0].set_ylabel('Значение ошибки')
axs[1,0].set_ylabel('Значение ошибки')

print('-'*30  + ' Train dataset ' + '-'*30)
classify(train, denormalize(train, transform), num_per_row=num_per_row, single_size=single_size, vspace=vspace,
            labels = train.classes, model=tunned_model, model_labels=train.classes, debug=True)
print('-'*30  + ' Test dataset ' + '-'*30)
classify(test, denormalize(test, transform), num_per_row=num_per_row, single_size=single_size, vspace=vspace,
            labels = test.classes, model=tunned_model, model_labels=test.classes)