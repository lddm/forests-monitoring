import os

import torch

import numpy as np
import pandas as pd
import telluric as tl
import matplotlib.pyplot as plt

from tqdm import tqdm
from torch import nn
from torch.autograd import Variable
from torch.utils.data import Dataset


from PIL import Image
from src.utils import preprocess_raster_image
from torchvision import transforms, models
from sklearn.preprocessing import MultiLabelBinarizer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#%% Constants
TRAIN_DIR = 'train/train-jpg'
MODEL_PATH = os.path.join('planet_challenge_model_one_cycle_lr_v2.tar')
DATA_PATH = 'data/high_res_Para'
DATASET_NAME = 'analytic_2016-06_2016-11_mosaic'
OUTPUT_FOLDER = os.path.join(DATA_PATH, 'classification')


#%% Auxiliary functions
def imshow(inp, fig_size=4, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)

    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    ax.imshow(inp)

    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


def initialize_resnet(num_classes, use_pretrained=True):
    resnet_model = models.resnet50(pretrained=use_pretrained)
    # Adjust last fully connected layer to number of classes in the
    # PlanetAmazonChallenge
    num_features = resnet_model.fc.in_features
    resnet_model.fc = nn.Linear(num_features, num_classes)
    return resnet_model


def classify(model, input_tensor, multi_label_binarizer, show_images=False, threshold=0):
    was_training = model.training
    model.eval()
    with torch.no_grad():
        inputs = input_tensor.to(device)
        outputs = model(inputs)
        preds = outputs > threshold
        # recover categorical labels from binary predictions
        labels = multi_label_binarizer.inverse_transform(preds.cpu())

        # labels is a list of size classification batch where each element
        # is a tuple with the labels corresponding to each image. The
        # submission expects the labels of each image to be outputted as a
        # space separated list
        output_labels = [' '.join(labels) for labels in labels]

        if show_images:
            fig = plt.figure(figsize=(10, 10))
            print('Labels: ', labels)
            imshow(inputs.squeeze(0).cpu().data)

    model.train(mode=was_training)
    return output_labels


def image_loader(image_array, transforms):
    """load image, returns cuda tensor"""
    # image = Image.open(image_name)
    image = Image.fromarray(image_array)
    image = transforms(image).float()
    image = Variable(image, requires_grad=True)
    image = image.unsqueeze(0)  # this is for VGG, may not be needed for ResNet
    # return image.cuda()  # assumes that you're using GPU
    return image


def load_classifier(dataset):
    model = initialize_resnet(len(dataset.mlb.classes_))
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters())

    checkpoint = torch.load(MODEL_PATH, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return model


def classify_raster(raster, model, transforms, multi_label_binarizer):
    image = preprocess_raster_image(raster)
    input_tensor = image_loader(image, transforms)
    labels = classify(model, input_tensor, multi_label_binarizer, show_images=False, threshold=0.2)
    geo_feature = tl.GeoFeature(raster.footprint(), {'labels': labels[0]})

    return geo_feature


class KaggleAmazonDataset(Dataset):
    """Dataset wrapping images and target labels for Kaggle - Planet Amazon from Space competition.

    Arguments:
        A CSV file path
        Path to image folder
        Extension of images
        Transform (optional) object containing transformations to apply on imagery.
    """

    def __init__(self, csv_path, img_path, img_ext, transform=None, number_samples=None, check_corrupt_files=True):

        self.tmp_df = pd.read_csv(csv_path)

        self.mlb = MultiLabelBinarizer()
        self.img_path = img_path
        self.img_ext = img_ext
        self.transform = transform

        if check_corrupt_files:
            self._check_corrupt_files()

        image_names = self.tmp_df['image_name']
        tags = self.tmp_df['tags']
        if number_samples:
            image_names = image_names[:number_samples]
            tags = tags[:number_samples]
            self.dataset_size = number_samples
        else:
            self.dataset_size = len(image_names)

        self.X_train = image_names
        # self.y_train is a sparse-matrix of size num_samples x num_classes where an element [i,j] equals 1
        # iff the sample with index 'i' correspond to class 'j'
        self.y_train = self.mlb.fit_transform(tags.str.split()).astype(np.float32)

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_path, self.X_train[index] +
                                      self.img_ext))
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        label = torch.from_numpy(self.y_train[index])
        return img, label

    def _check_corrupt_files(self):
        # check that all images listed in the train.csv are available on the training folder
        assert self.tmp_df['image_name'].apply(lambda x: os.path.isfile(os.path.join(
            self.img_path, x + self.img_ext))).all(), \
            "Some images referenced in the CSV file were not found"

        # some files available in the folder are corrupted causing an PIL.UnidentifiedImageError
        for image_name in self.tmp_df['image_name']:
            file_size = os.stat(os.path.join(
                self.img_path, image_name + self.img_ext)).st_size
            if file_size == 0:
                raise (OSError('File {} is corrupt'.format(image_name)))

    def decode_binary_label(self, array):
        return self.mlb.inverse_transform(array)

    def __len__(self):
        return len(self.X_train.index)


#%% main
if __name__ == "__main__":
    test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = KaggleAmazonDataset('train_v2.csv', TRAIN_DIR, '.jpg', check_corrupt_files=False)
    model = load_classifier(dataset)

    # Process raster by chunks of size 224x224 pixels.
    geo_features_list = []
    chunk_size = 224

    # Load one raster to estimate number of chunks that will be processed per each raster
    rasters_path = os.path.join(DATA_PATH, DATASET_NAME)
    raster_path = os.path.join(rasters_path, os.listdir(rasters_path)[0])
    raster = tl.GeoRaster2.open(raster_path)
    raster_size = max(raster.shape)  # this assumes raster width and height are equal
    number_of_chunks = int(np.ceil(raster_size / chunk_size) ** 2)

    rasters = os.listdir(rasters_path)
    number_of_rasters = len(rasters)

    with tqdm(total=number_of_rasters * number_of_chunks) as pbar:
        for raster_filename in rasters:
            if raster_filename.endswith(".tif"):
                # Load raster
                raster_path = os.path.join(rasters_path, raster_filename)
                raster = tl.GeoRaster2.open(raster_path)

                for chunk in raster.chunks(chunk_size):
                    raster = chunk.raster
                    geo_features_list.append(classify_raster(raster, model, test_transforms, dataset.mlb))
                    pbar.update(1)
            else:
                continue

    feature_collection = tl.FeatureCollection(geo_features_list)
    output_path = os.path.join(OUTPUT_FOLDER, DATASET_NAME + '.geojson')
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    feature_collection.save(output_path)
