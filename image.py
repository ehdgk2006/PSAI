from simulator import make_frames

import os
import albumentations
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torch


class ImagePaths(Dataset):
    def __init__(self, path, size=None):
        self.size = size

        self.images = [os.path.join(path, file) for file in os.listdir(path)]
        self._length = len(self.images)

        self.rescaler = albumentations.SmallestMaxSize(max_size=self.size)
        self.cropper = albumentations.CenterCrop(height=self.size, width=self.size)
        self.preprocessor = albumentations.Compose([self.rescaler, self.cropper])


    def __len__(self):
        return self._length


    def preprocess_image(self, image_path):
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        image = self.preprocessor(image=image)["image"]
        image = (image / 255).astype(np.float32)
        image = image.transpose(2, 0, 1)
        return image


    def __getitem__(self, i):
        example = self.preprocess_image(self.images[i])
        return example
    

class EncodedDataset(Dataset):
    def __init__(self, path, img_size=None):
        self.img_size = img_size

        self.rescaler = albumentations.SmallestMaxSize(max_size=self.img_size)
        self.cropper = albumentations.CenterCrop(height=self.img_size, width=self.img_size)
        self.preprocessor = albumentations.Compose([self.rescaler, self.cropper])

        self.images = []
        self.path = path

        self._length = len(self.images) // 4


    def load(self, arr):
        self.images = arr
        self._length = len(self.images) // 4


    def __len__(self):
        return self._length
    
    
    def load_data(self, model=None):
        def encode(x):
            h = model.encoder(x)
            h = model.quant_conv(h)
            quant, _, _ = model.quantize(h)
            return quant

        with torch.no_grad():
            for j in range(len(self.images), 4000):
                for i, file in enumerate(os.listdir(self.path), start=1):
                    if file == f'image_{j:05d}.jpg':
                        image = self.preprocess_image(os.path.join(self.path, file))
                        image = torch.FloatTensor(image).reshape(1, 3, 256, 256).to(device)

                        latent_vector = encode(image).squeeze()
                        self.images.append(latent_vector)
                        break
                if (j % 100 == 99):
                    torch.save(self.images, './circle.ds')
        
        self._length = len(self.images) // 4


    def preprocess_image(self, image_path):
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        image = self.preprocessor(image=image)["image"]
        image = (image / 255).astype(np.float32)
        image = image.transpose(2, 0, 1)
        return image


    def __getitem__(self, i):
        example = self.images[i*4:i*4+4]
        return example


def make_encoded_images(images, img_size, model):
    def encode(x):
        h = model.encoder(x)
        h = model.quant_conv(h)
        quant, _, _ = model.quantize(h)
        return quant
    
    device = model.device
    rescaler = albumentations.SmallestMaxSize(max_size=img_size)
    cropper = albumentations.CenterCrop(height=img_size, width=img_size)
    preprocessor = albumentations.Compose([rescaler, cropper])

    latent_vectors = []
    res = []

    with torch.no_grad():
        for i in range(len(images)):
            image = images[i]
            image = np.array(image).astype(np.uint8)
            image = preprocessor(image=image)["image"]
            image = (image / 255).astype(np.float32)
            image = image.transpose(2, 0, 1)

            image = torch.FloatTensor(image).reshape(1, 3, 256, 256).to(device)
            # image += torch.randn_like(image) * 0.05
            latent_vector = encode(image)

            latent_vectors.append(latent_vector)
            res.append(image)
    return res, latent_vectors


def make_data(model, obj, img_size, batch_size = 8, frames = 10):
    batch = []
    encoded_batch = []
    
    data = make_frames(obj, frames)
    data, encoded_data = make_encoded_images(data, img_size, model)
    
    for i in range(len(data)):
        batch.append([data[i]])
        encoded_batch.append([encoded_data[i]])
    
    for i in range(batch_size-1):
        data = make_frames(obj, frames)
        data, encoded_data = make_encoded_images(data, img_size, model)
        
        for j in range(len(data)):
            batch[j].append(data[j])
            encoded_batch[j].append(encoded_data[j])
    
    for i in range(len(batch)):
        batch[i] = torch.cat(batch[i], dim=0)
        encoded_batch[i] = torch.cat(encoded_batch[i], dim=0)
    
    return batch, encoded_batch


if __name__ == '__main__':
    device = torch.device("cuda")
    # film = yaml.load(open('./configs/model.yaml'), Loader=yaml.FullLoader)
    # model = VQModel(film['model']['params']['ddconfig'], film['model']['params']['lossconfig'], film['model']['params']['n_embed'], film['model']['params']['embed_dim'], "./ckpts/last.ckpt").to(device)
    # model.eval()

    dataset = EncodedDataset('./data/circle', img_size=256)
    # dataset.load_data(model)
    dataset.load(torch.load('./circle.ds'))
    print(len(dataset.images))
