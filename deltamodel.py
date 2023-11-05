"""
pip install opencv-python
pip install imageio
pip install albumentations
pip install lightning
pip install einops
apt-get update && apt-get install libgl1
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import utils as vutils

from image import make_data
from losses.lpips import LPIPS
from vqgan import VQModel
from simulator import Circle, Pendulum

import os
import math
import yaml
import random
from PIL import Image
import imageio


def DeltaLoss(Y_pred: torch.Tensor, Y: torch.Tensor, last_img, pred_img, target_img, lpips: LPIPS):
    y_flat = torch.flatten(Y, 1)
    y_pred_flat = torch.flatten(Y_pred, 1)

    perceptual_loss = lpips(pred_img, target_img)
    rec_loss = torch.abs(pred_img - target_img)
    perceptual_rec_loss = perceptual_loss + rec_loss
    perceptual_rec_loss = perceptual_rec_loss.mean()
    
    x_perceptual_loss = lpips(pred_img, last_img)
    x_rec_loss = torch.abs(pred_img - last_img)
    x_perceptual_rec_loss = x_perceptual_loss + x_rec_loss
    x_perceptual_rec_loss = x_perceptual_rec_loss.mean()
    
    encoded_l1_loss = F.l1_loss(y_pred_flat, y_flat)

    loss = perceptual_rec_loss + encoded_l1_loss + 0.2*(x_perceptual_rec_loss - 0.75) ** 2

    return loss


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 10):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        x = x + self.pe[0, :x.size(1)]
        return x


class DeltaBlock(nn.Module):
    def __init__(self, in_features, out_features, is_activation=True):
        super(DeltaBlock, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.is_activation = is_activation

        self.model = nn.Sequential(
            nn.Linear(in_features, out_features),
        )
    

    def forward(self, x):
        z = self.model(x)
        if self.in_features == self.out_features:
            z = z + x
        
        if self.is_activation:
            z = nn.functional.gelu(z)
        
        return z


class DeltaModel(nn.Module):
    def __init__(self, in_shape, out_shape):
        super(DeltaModel, self).__init__()
        
        self.comp_channel = 128
        self.in_shape = in_shape        # (c, w, h)
        self.out_shape = out_shape      # (c, w, h)
        self.in_features = self.comp_channel * in_shape[1] * in_shape[2]
        self.out_features = self.comp_channel * out_shape[1] * out_shape[2]
        
        self.conv1 = nn.Conv2d(in_shape[0], self.comp_channel, 1)
        self.conv2 = nn.Conv2d(self.comp_channel, out_shape[0], 1)

        self.unflat = nn.Unflatten(1, (self.comp_channel, out_shape[1], out_shape[2]))
        self.flat = nn.Flatten()

        self.pe = PositionalEncoding(self.in_features)
        self.attn = nn.MultiheadAttention(self.in_features, 4, batch_first=True)

        self.model = nn.Sequential(
            nn.LayerNorm((self.in_features)),
            DeltaBlock(self.in_features, 64),
            DeltaBlock(64, 64),
            nn.LayerNorm((64)),
            DeltaBlock(64, 64),
            DeltaBlock(64, self.in_features, False),
            nn.LayerNorm((self.in_features)),
        )
    

    def attention(self, x):
        q = x[0]
        k = x[1]
        v = x[2]
        
        return self.attn(q, k, v, need_weights=False)[0]


    def forward(self, x):
        z = []
        for i in range(len(x)):
            temp = self.conv1(x[i])
            temp = self.flat(temp).unsqueeze(1)
            z.append(temp)

        q = z[-1]
        z = torch.cat(z[:len(z)], 1)

        # q = self.pe(q)
        z = self.pe(z)

        z = self.attention([q, z, z])
        z = (self.model(z)).squeeze(1)
        z = self.unflat(z)
        z = self.conv2(z)
        z = z*nn.functional.sigmoid(z)
        return z


def test_model(vqmodel, deltamodel, epoch):
    encoded_frames = []
    encoded_frames += make_data(vqmodel, Circle(), 256, 1, 10)[1]

    decoded_frames = []

    with torch.no_grad():
        for i in range(30):
            x = encoded_frames[max(1, len(encoded_frames)-10):]

            next_encoded_frame = deltamodel(x)
            next_encoded_frame = vqmodel.quant_conv(next_encoded_frame)
            next_encoded_frame, _, _ = vqmodel.quantize(next_encoded_frame)
            encoded_frames.append(next_encoded_frame)

        for i in range(30):
            z = vqmodel.decode(encoded_frames[i])

            grid = vutils.make_grid(z)
            # Add 0.5 after unnormalizing to [0, 255] to round to the nearest integer
            ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
            im = Image.fromarray(ndarr)
            decoded_frames.append(im)

    imageio.mimsave(f'./saves/test_{epoch}.gif', decoded_frames, duration=50)
    print('end')
    

if __name__ == '__main__':
    device = torch.device('cpu')

    film = yaml.load(open('./configs/model.yaml'), Loader=yaml.FullLoader)
    vqmodel = VQModel(film['model']['params']['ddconfig'], film['model']['params']['lossconfig'], film['model']['params']['n_embed'], film['model']['params']['embed_dim'], "./ckpts/last.ckpt").to(device)
    vqmodel.eval()

    lpips = LPIPS().to(device)
    lpips.eval()

    # dataset = EncodedDataset('./data/circle', img_size=256)
    # dataset.load(torch.load('./circle.ds'))
    # dataset = DataLoader(dataset, batch_size=8, shuffle=True)

    model = DeltaModel((256, 16, 16), (256, 16, 16)).to(device)
    # model.load_state_dict(torch.load("./saves/deltamodel_circle.pth"))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, betas=(0.5, 0.9))
    criterion = DeltaLoss

    EPISODES = 500
    running_loss = 0.
    last_loss = 0.

    for episode in range(EPISODES):
        obj = Circle()
        data, encoded_data = make_data(vqmodel, obj, 256, 8, random.randint(3, 11))

        inputs, labels = encoded_data[:len(encoded_data)-1], encoded_data[-1]

        optimizer.zero_grad()

        outputs = model(inputs)
        outputs = vqmodel.quant_conv(outputs)
        q, _, _ = vqmodel.quantize(outputs)
        
        last_img = data[-2]
        pred_img = vqmodel.decode(q)
        target_img = data[-1]
        
        loss = criterion(outputs, labels, last_img, pred_img, target_img, lpips)
        loss.backward()

        optimizer.step()

        running_loss += loss.item()
        
        if episode % 5 == 4:
            last_loss = running_loss / 5. # loss per 5 episodes
            print('EPISODE {}, loss: {}'.format(episode + 1, last_loss))
            running_loss = 0.

        if episode % 100 == 99:
            # os.remove('./saves/deltamodel_circle.pth')
            # torch.save(model.state_dict(), './saves/deltamodel_circle.pth')
            test_model(vqmodel, model, episode)
