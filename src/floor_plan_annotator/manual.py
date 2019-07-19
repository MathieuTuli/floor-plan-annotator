''' Manual Annotation tool
'''

from pathlib import Path

import numpy as np
import cv2


class ManualAnnotator:
    def __init__(self,):
        '''
        '''
        pass

    def run(self,):
        houses_folder = 'sydney-house/rent_crawler/goodhouses'
        for house_folder in houses_folder.iterdir():
            for house in house_folder.iterdir():
                if 'floorplan_label' in str(house) and house.suffix == '.png':
                    print(house)
