'''
* This automatic annotator is fully based off of the work done by CubCasa
* Their source code can be found at https://github.com/CubiCasa/CubiCasa
* date: 2019-07-11
* author: MathieuTuli
'''
from pathlib import Path
from typing import Tuple, List, Any

from floortrans.post_prosessing import split_prediction, get_polygons, \
    split_validation
from mpl_toolkits.axes_grid1 import AxesGrid
from floortrans.plotting import segmentation_plot, polygons_to_image, \
    draw_junction_from_dict, discrete_cmap
from floortrans.loaders import FloorplanSVG, DictToTensor, Compose, \
    RotateNTurns
from floortrans.models import get_model
from torch.utils.data import DataLoader
from skimage import transform

import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import torch
import cv2


class AutomaticAnnotator:
    def __init__(self, checkpoint_path: Path):
        self.rot, self.room_classes, self.icon_classes, self.n_classes, \
            self.split, self.model = self.load_model(checkpoint_path)
        self.n_rooms = len(self.room_classes)
        self.n_icons = len(self.icon_classes)

    def load_model(self, model_path: str) -> Tuple(Any, List, List,
                                                   int, List, Any):
        discrete_cmap()
        rot = RotateNTurns()
        room_classes = ["Background", "Outdoor", "Wall", "Kitchen",
                        "Living Room", "Bed Room", "Bath", "Entry", "Railing",
                        "Storage", "Garage", "Undefined"]
        icon_classes = ["No Icon", "Window", "Door", "Closet",
                        "Electrical Applience", "Toilet", "Sink",
                        "Sauna Bench", "Fire Place", "Bathtub", "Chimney"]
        model = get_model('hg_furukawa_original', 51)
        n_classes = 44
        split = [21, 12, 11]
        model.conv4_ = torch.nn.Conv2d(256, n_classes,
                                       bias=True, kernel_size=1)
        model.upsample = torch.nn.ConvTranspose2d(n_classes, n_classes,
                                                  kernel_size=4, stride=4)
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state'])
        model.eval()
        model.cuda()
        return (rot, room_classes, icon_classes, n_classes, split, model)

    def annotate_image(self, image_path: str):
        with torch.no_grad():
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            height, width, _ = img.shape
            img_size = (height, width)
            img = np.moveaxis(img, -1, 0)
            img = torch.tensor(img.astype(np.float32))
            img = img.cuda()

            rotations = [(0, 0), (1, -1), (2, 2), (-1, 1)]
            pred_count = len(rotations)
            prediction = torch.zeros(
                [pred_count, self.n_classes, height, width])
            for i, r in enumerate(rotations):
                forward, back = r
                # We rotate first the image
                rot_image = self.rot(img, 'tensor', forward)
                pred = self.model(rot_image)
                # We rotate prediction back
                pred = self.rot(pred, 'tensor', back)
                # We fix heatmaps
                pred = self.rot(pred, 'points', back)
                # We make sure the size is correct
                pred = F.interpolate(pred, size=(
                    height, width), mode='bilinear', align_corners=True)
                # We add the prediction to output
                prediction[i] = pred[0]

            prediction = torch.mean(prediction, 0, True)

            rooms_pred = F.softmax(
                prediction[0, 21:21+12], 0).cpu().data.numpy()
            rooms_pred = np.argmax(rooms_pred, axis=0)

            icons_pred = F.softmax(prediction[0, 21+12:], 0).cpu().data.numpy()
            icons_pred = np.argmax(icons_pred, axis=0)

            plt.figure(figsize=(12, 12))
            ax = plt.subplot(1, 1, 1)
            ax.axis('off')
            rseg = ax.imshow(rooms_pred, cmap='rooms',
                             vmin=0, vmax=self.n_rooms-0.1)
            cbar = plt.colorbar(rseg, ticks=np.arange(
                self.n_rooms) + 0.5, fraction=0.046, pad=0.01)
            cbar.ax.set_yticklabels(self.room_classes, fontsize=20)
            plt.show()

            plt.figure(figsize=(12, 12))
            ax = plt.subplot(1, 1, 1)
            ax.axis('off')
            iseg = ax.imshow(icons_pred, cmap='icons',
                             vmin=0, vmax=self.n_icons-0.1)
            cbar = plt.colorbar(iseg, ticks=np.arange(
                self.n_icons) + 0.5, fraction=0.046, pad=0.01)
            cbar.ax.set_yticklabels(self.icon_classes, fontsize=20)
            plt.show()

            heatmaps, rooms, icons = split_prediction(prediction, img_size,
                                                      self.split)
            polygons, types, room_polygons, room_types = get_polygons(
                (heatmaps, rooms, icons), 0.2, [1, 2])

            pol_room_seg, pol_icon_seg = polygons_to_image(
                polygons, types, room_polygons, room_types, height, width)
            plt.figure(figsize=(12, 12))
            ax = plt.subplot(1, 1, 1)
            ax.axis('off')
            rseg = ax.imshow(pol_room_seg, cmap='rooms',
                             vmin=0, vmax=self.n_rooms-0.1)
            cbar = plt.colorbar(rseg, ticks=np.arange(
                self.n_rooms) + 0.5, fraction=0.046, pad=0.01)
            cbar.ax.set_yticklabels(self.room_classes, fontsize=20)
            plt.tight_layout()
            plt.show()

            plt.figure(figsize=(12, 12))
            ax = plt.subplot(1, 1, 1)
            ax.axis('off')
            iseg = ax.imshow(pol_icon_seg, cmap='icons',
                             vmin=0, vmax=self.n_icons-0.1)
            cbar = plt.colorbar(iseg, ticks=np.arange(
                self.n_icons) + 0.5, fraction=0.046, pad=0.01)
            cbar.ax.set_yticklabels(self.icon_classes, fontsize=20)
            plt.tight_layout()
            plt.show()


if __name__ == "__main__":
    annotator = AutomaticAnnotator('cubicasa_model.pkl')
    annotator.annotate_image('test.png')
