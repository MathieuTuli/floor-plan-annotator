''' Manual Annotation tool
'''

from pathlib import Path
from typing import List, Tuple

import sys

import numpy as np
import cv2

from .components import CornerAnnotation, WindowAnnotation, \
    DoorAnnotation, EdgeAnnotation, GraphAnnotations

CLICK_BARRIER = False
REF_PTS = list()


class ManualAnnotator:
    def __init__(self, floor_plans: List[Path], save_to: Path):
        '''
        '''
        self.floor_plans = floor_plans
        self.save_to: Path = save_to
        self.build_directories()
        self.current_graph: GraphAnnotations = None

    @staticmethod
    def record_click(event, x, y, flags, param):
        global REF_PTS
        global CLICK_BARRIER
        CLICK_BARRIER = False
        if event == cv2.EVENT_LBUTTONDOWN:
            REF_PTS = [(x, y)]
            CLICK_BARRIER = True
            print("CLICK")
        elif event == cv2.EVENT_LBUTTONUP:
            REF_PTS.append((x, y))
            CLICK_BARRIER = True

    def build_directories(self):
        if not self.save_to.is_dir():
            self.save_to.mkdir(parents=True)
        children = ['floor-plans']
        for child in children:
            new_folder = self.save_to / child
            if not new_folder.is_dir():
                new_folder.mkdir(parents=True)

    def write(self, annotation: GraphAnnotations) -> bool:
        raise NotImplementedError

    def draw_point(self, point: Tuple[int, int],
                   colour: Tuple[int, int, int],
                   image: np.ndarray) -> np.ndarray:
        cv2.circle(image, point, colour)
        return image

    def draw_line(self, point_a: Tuple[int, int], point_b: Tuple[int, int],
                  colour: Tuple[int, int, int],
                  image: np.ndarray) -> np.ndarray:
        return image

    def get_key(self, message: str) -> str:
        '''returns 0 for not confirmed, 1 for confirmed, -1 for quit
        '''
        print(message)
        print("Press [q] to quit.")
        key = cv2.waitKey(0)
        return key & 0xFF

    def wait_for_click(self):
        print('Waiting for a click')
        global CLICK_BARRIER
        while not CLICK_BARRIER:
            continue

    def add_corner(self, img: np.ndarray) -> CornerAnnotation:
        img_copy = img.copy()
        self.wait_for_click()
        while True:
            global REF_PTS
            if len(REF_PTS) == 2:
                self.draw_point(img_copy, REF_PTS[0], (0, 255, 255))

    def add_window(self, img: np.ndarray) -> WindowAnnotation:
        img_copy = img.copy()
        global REF_PTS
        self.wait_for_click()
        if len(REF_PTS) == 2:
            self.draw_point(img_copy, REF_PTS[0], (0, 255, 0))

    def add_door(self, img: np.ndarray) -> DoorAnnotation:
        img_copy = img.copy()
        global REF_PTS
        self.wait_for_click()
        if len(REF_PTS) == 2:
            self.draw_point(img_copy, REF_PTS[0], (255, 0, 0))

    def add_edge(self, img: np.ndarray) -> EdgeAnnotation:
        img_copy = img.copy()
        global REF_PTS
        raise NotImplementedError

    def run(self,):
        for floor_plan in self.floor_plans:
            img = cv2.imread(str(floor_plan))
            if img is None:
                print(f"\n\n{floor_plan} could not be read.\n\n")
                continue
            # self.write(annotation)
            # if len(REF_PTS) == 2:
            #     cv2.rectangle(img, REF_PTS[0], REF_PTS[1])
            img_copy = img.copy()
            while True:
                global REF_PTS
                REF_PTS = list()
                cv2.destroyAllWindows()
                cv2.namedWindow("current_floor_plan")
                cv2.moveWindow("current_floor_plan", 20, 20)
                cv2.setMouseCallback("current_floor_plan",
                                     ManualAnnotator.record_click)
                cv2.imshow("current_floor_plan", img)
                key = self.get_key(
                    'What would you like to draw?\n' +
                    '[c] corner | [w] window | [d] door | [e] edge' +
                    '| [p] disregard image')
                if key == ord('c'):
                    # self.add_corner(img)
                    while True:
                        if len(REF_PTS) == 2:
                            print("YES")
                            REF_PTS = list()
                elif key == ord('w'):
                    self.add_window(img)
                elif key == ord('d'):
                    self.add_door(img)
                elif key == ord('e'):
                    self.add_edge(img)
                elif key == ord('p'):
                    pass
                elif key == ord('q'):
                    sys.exit(0)
                else:
                    print("unknown selection")
                    continue
                break


if __name__ == '__main__':
    houses_folder = Path('sydney-house/rent_crawler/goodhouses')
    houses = list()
    for house_folder in houses_folder.iterdir():
        for house in house_folder.iterdir():
            if 'floorplan_label' in str(house) and house.suffix == '.png':
                houses.append(house)

    annotator = ManualAnnotator(houses, Path('processed_houses'))
    annotator.run()
