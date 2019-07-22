'''
* This automatic annotator is fully based off of the work done by CubCasa
* Their source code can be found at https://github.com/CubiCasa/CubiCasa
* date: 2019-07-11
* author: MathieuTuli
'''
from argparse import ArgumentParser
from pathlib import Path
from typing import Tuple, List, Any
import importlib.resources
import sys


# from mpl_toolkits.axes_grid1 import AxesGrid
from torch.utils.data import DataLoader
from skimage import transform

import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import torch
import cv2

from .CubiCasa5k.floortrans.post_prosessing import split_prediction, \
    get_polygons, split_validation
from .CubiCasa5k.floortrans.plotting import segmentation_plot, \
    polygons_to_image, draw_junction_from_dict, discrete_cmap
from .CubiCasa5k.floortrans.loaders import FloorplanSVG, DictToTensor, \
    Compose, RotateNTurns
from .CubiCasa5k.floortrans.models import get_model

parser = ArgumentParser('')
parser.add_argument('--cubi', action='store_true')
parser.set_defaults(cubi=False)
args = parser.parse_args()


class AutomaticAnnotator:
    def __init__(self, checkpoint_path: Path):
        self.rot, self.room_classes, self.icon_classes, self.n_classes, \
            self.split, self.model = self.load_model(checkpoint_path)
        self.n_rooms = len(self.room_classes)
        self.n_icons = len(self.icon_classes)

    def load_model(self, model_path: str) -> Tuple[List, List,
                                                   int, List, Any]:
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

    def annotate_image(self, image: Any, height: int, width: int):
        img_size = (height, width)
        with torch.no_grad():
            rotations = [(0, 0), (1, -1), (2, 2), (-1, 1)]
            pred_count = len(rotations)
            prediction = torch.zeros(
                [pred_count, self.n_classes, height, width])
            for i, r in enumerate(rotations):
                forward, back = r
                # We rotate first the image
                rot_image = self.rot(image, 'tensor', forward)
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

            print("Doing rooms")
            plt.figure(figsize=(12, 12))
            ax = plt.subplot(1, 1, 1)
            ax.axis('off')
            rseg = ax.imshow(rooms_pred, cmap='rooms',
                             vmin=0, vmax=self.n_rooms-0.1)
            cbar = plt.colorbar(rseg, ticks=np.arange(
                self.n_rooms) + 0.5, fraction=0.046, pad=0.01)
            cbar.ax.set_yticklabels(self.room_classes, fontsize=20)
            plt.show()

            print("Doing icons")
            plt.figure(figsize=(12, 12))
            ax = plt.subplot(1, 1, 1)
            ax.axis('off')
            iseg = ax.imshow(icons_pred, cmap='icons',
                             vmin=0, vmax=self.n_icons-0.1)
            cbar = plt.colorbar(iseg, ticks=np.arange(
                self.n_icons) + 0.5, fraction=0.046, pad=0.01)
            cbar.ax.set_yticklabels(self.icon_classes, fontsize=20)
            plt.show()

            print("Doing post")
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

    def confirm(self,):
        pass


def normalize_to_origin(nodes,
                        img) -> Tuple[List[Tuple[int, int]], np.ndarray]:
    pass


def get_list_of_non_zero(mask) -> List[Tuple[int, int]]:
    nodes = cv2.findNonZero(mask)
    nodes = [(int(node[0][0]), int(node[0][1]))
             for node in nodes]
    return nodes


def conform_nodes(nodes: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    threshold = 9
    other_x = 0
    other_y = 0


def yellow_corners(
        img: np.ndarray) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    yellow = [0, 255, 255]
    mask = np.array(yellow)
    mask = cv2.inRange(img, mask, mask)
    for i, row in enumerate(mask):
        for j, col in enumerate(row):
            if col == 255:
                try:
                    # first left coord, block out the others
                    range_ = [-3, -2, -1, 1, 0,
                              1, 3, 4, 5, 6, 7, 8, 9]
                    for x in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
                        for y in range_:
                            mask[i + x][j + y] = 0
                    mask[i][j] = 255
                except Exception:
                    pass
    nodes = get_list_of_non_zero(mask)
    print(nodes)
    nodes = conform_nodes(nodes)
    print(nodes)
    return mask, nodes


def green_windows(
        img: np.ndarray,
        corners: List[Tuple[int, int]]) -> Tuple[np.ndarray,
                                                 List[Tuple[int, int]]]:
    green = [0, 255, 0]
    mask = np.array(green)
    mask = cv2.inRange(img, mask, mask)
    nodes = get_list_of_non_zero(mask)
    for i, row in enumerate(mask):
        for j, col in enumerate(row):
            pass
    nodes = get_list_of_non_zero(mask)
    return mask, nodes


def blue_doors(
        img: np.ndarray,
        corners: List[Tuple[int, int]]) -> Tuple[np.ndarray,
                                                 List[Tuple[int, int]]]:
    blue = [255, 0, 0]
    mask = np.array(blue)
    mask = cv2.inRange(img, mask, mask)
    for i, row in enumerate(mask):
        pass
    nodes = get_list_of_non_zero(mask)
    return mask, nodes


def red_walls(
        img: np.ndarray,
        corners: List[Tuple[int, int]]) -> Tuple[np.ndarray,
                                                 List[Tuple[int, int]]]:
    red = [0, 0, 255]
    mask = np.array(red)
    mask = cv2.inRange(img, mask, mask)
    for i, row in enumerate(mask):
        pass
    nodes = get_list_of_non_zero(mask)
    return mask, nodes


def annotate_from_colors(houses_folder: Path):
    np.set_printoptions(threshold=sys.maxsize)
    # ORDER MATTERS | Y THEN B THEN G
    for house_folder in houses_folder.iterdir():
        for house in house_folder.iterdir():
            if 'floorplan_label' in str(house) and house.suffix == '.png':
                print(house)
                img = cv2.imread(str(house))
                # B G R
                # Check out this janky shit
                # now i need to align these janky corners
                yellow_mask, corners = yellow_corners(img)
                blue_mask, windows = green_windows(img, corners)
                green_mask, doors = blue_doors(img, corners)
                red_mask, walls = red_walls(img, corners)
                cv2.imshow('yellow', yellow_mask)
                cv2.imshow('green', green_mask)
                cv2.imshow('blue', blue_mask)
                cv2.imshow('red', red_mask)
                cv2.waitKey(0)
                sys.exit(0)


if __name__ == "__main__":
    if args.cubi:
        model_path = importlib.resources.path(
            'floor_plan_annotator.models',
            'cubicasa_model.pkl')
        annotator = AutomaticAnnotator(str(next(model_path.gen)))
        # image_list_path = importlib.resources.path(
        #     'floor_plan_annotator.data',
        #     'test.txt')
        # image_path = str(next(image_list_path.gen))
        # print(image_path)
        dataset = FloorplanSVG('data/sydney-house/', 'predict.txt',
                               format='txt', original_size='True')
        data_loader = DataLoader(dataset, batch_size=1, num_workers=0)
        data_iter = iter(data_loader)
        for item in data_iter:
            item = next(data_iter)
            image = item['image'].cuda()
            _, _, h, w = image.shape
            annotator.annotate_image(image, h, w)
    else:
        annotate_from_colors(Path('sydney-house/rent_crawler/goodhouses'))
