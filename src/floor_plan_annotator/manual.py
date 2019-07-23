# TODO: Make Frame class that has frame: np.ndarray and annotions: List[str] as
# @ members that can handle all the drawing on its own
from pathlib import Path
from typing import Tuple, List

import numpy as np
import cv2
import sys

from python_shape_grammars.floor_plan_elements import Node, Edge, RoomNode

from .components import CornerAnnotation, WindowAnnotation, \
    DoorAnnotation, EdgeAnnotation, GraphAnnotations

REF_PTS = list()
PREV_EDGE_NODE = None
CLICK_BARRIER = False


class ManualAnnotator:
    def __init__(self,
                 floor_plans: List[Path],
                 save_to: Path,):
        self.floor_plans = floor_plans
        self.save_to: Path = save_to
        self.build_directories()
        self.current_graph: GraphAnnotations = None
        self.corners = list()
        self.windows = list()
        self.doors = list()
        self.walls = list()

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

    def write_results(self, results: GraphAnnotations) -> None:
        image_path = self.image_bank / f"{self.frame_counter}.jpg"
        annotation_path = self.annotation_file_bank / \
            f"{self.frame_counter}.txt"
        cv2.imwrite(str(image_path), results.frame)
        with open(annotation_path, "w") as f:
            image_width = results.image_width
            image_height = results.image_height
            f.write("{class_name},{left},{top},{right}," +
                    "{bottom},{image_width},{image_height}\n")
            for obj in results.objects:
                f.write(f"{obj.class_name},{obj.bbox.left},{obj.bbox.top}" +
                        f",{obj.bbox.right},{obj.bbox.bottom},{image_width}," +
                        f"{image_height}\n")
        self.frame_counter += 1

    def run(self,) -> None:
        for floor_plan in self.floor_plans:
            frame = cv2.imread(str(floor_plan))
            frame_copy = frame.copy()
            while True:
                cv2.destroyAllWindows()
                cv2.namedWindow("image")
                cv2.moveWindow("image", 20, 20)
                cv2.imshow("image", frame_copy)
                print('What would you like to draw?\n' +
                      '[c] corner | [w] window | [d] door | [e] edge' +
                      '| [p] disregard image\n' +
                      'Press [q] to quit fully\n' +
                      'Press [n] to move on')
                key = cv2.waitKey(0)
                key = key & 0xFF
                if key == ord('c') or key == ord('w') \
                        or key == ord('d') or key == ord('e'):
                    frame, results = self.annotate(frame_copy, key)
                elif key & 0xFF == ord('q'):
                    print("Are you sure you want to quit [y]/[n]")
                    while True:
                        q = cv2.waitKey(0)
                        if q & 0xFF == ord('y'):
                            sys.exit(0)
                        elif q & 0xFF == ord('n'):
                            break
                elif key & 0xFF == ord('n'):
                    print("Are you sure you want to move on")
                    _continue = False
                    while True:
                        q = cv2.waitKey(0)
                        if q & 0xFF == ord('y'):
                            _continue = True
                            break
                        elif q & 0xFF == ord('n'):
                            break
                    if _continue:
                        break

    def draw_point(self, image: np.ndarray, point: Tuple[int, int],
                   colour: Tuple[int, int, int],
                   ) -> np.ndarray:
        cv2.circle(image, center=point, radius=1, color=colour)
        return image

    def add_corner(self, points: List[Tuple[int, int]]):
        cur_x, cur_y = points[0]
        if len(self.corners) == 0:
            self.corners.append((cur_x, cur_y))
            return
        prev_x, prev_y = self.corners[-1]
        if abs(cur_x - prev_x) < abs(cur_y - prev_y):
            self.corners.append((prev_x, cur_y))
        else:
            self.corners.append((cur_x, prev_y))

    def add_window(self, points: List[Tuple[int, int]]):
        pass

    def add_door(self, points: List[Tuple[int, int]]):
        pass

    def add_wall(self, points: List[Tuple[int, int]]):
        pass

    def annotate(self, frame: np.ndarray,
                 _type: str) -> Tuple[np.ndarray, str]:
        cv2.destroyAllWindows()
        cv2.namedWindow("clickable_image")
        cv2.moveWindow("clickable_image", 20, 20)
        cv2.setMouseCallback("clickable_image",
                             ManualAnnotator.record_click)
        old_frame = frame.copy()
        if _type == ord('c'):
            print("CORNER")
        elif _type == ord('w'):
            print("WINDOW")
        elif _type == ord('d'):
            print("DOOR")
        elif _type == ord('e'):
            print("WALL")
        print("Press [q] at any time to quit this new drawing session")
        while True:
            global REF_PTS
            if len(REF_PTS) == 2:
                if _type == ord('c'):
                    cv2.circle(frame, REF_PTS[0], radius=4, color=(0, 0, 255))
                elif _type == ord('w'):
                    cv2.circle(frame, REF_PTS[0], radius=4, color=(0, 0, 255))
                elif _type == ord('d'):
                    cv2.circle(frame, REF_PTS[0], radius=4, color=(0, 0, 255))
                elif _type == ord('e'):
                    cv2.circle(frame, REF_PTS[0], radius=4, color=(0, 0, 255))
                cv2.imshow('clickable_image', frame)
                print("Press [y] to confirm, else press any other key.")
                print("Press [q] to cancel this operation")
                key = cv2.waitKey(0)
                if key & 0xFF == ord('y'):
                    print("You press [y], creating new annotation")
                    old_frame = frame
                    if _type == ord('c'):
                        self.add_corner(REF_PTS)
                    elif _type == ord('w'):
                        self.add_window(REF_PTS)
                    elif _type == ord('d'):
                        self.add_door(REF_PTS)
                    elif _type == ord('e'):
                        self.add_wall(REF_PTS)
                elif key & 0xFF == ord('q'):
                    print("Quitting new session")
                    return frame, None
                else:
                    frame = old_frame.copy()
                REF_PTS = list()
            else:
                cv2.imshow('clickable_image', frame)
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    print("Quitting session")
                    return frame, None

        # This would happen if you click right and drag left to draw
        # drawn rt to lb
        # if REF_PTS[0][0] > REF_PTS[1][0] and REF_PTS[0][1] < REF_PTS[1][1]:
        #     rt = REF_PTS[0]
        #     lb = REF_PTS[1]
        #     REF_PTS[0] = (lb[0], rt[1])
        #     REF_PTS[1] = (rt[0], lb[1])
        # # drawn rb to lt
        # elif REF_PTS[0][0] > REF_PTS[1][0] and REF_PTS[0][1] > REF_PTS[1][1]:
        #     temp = REF_PTS[0]
        #     REF_PTS[0] = REF_PTS[1]
        #     REF_PTS[1] = temp
        # # drawn lb to rt
        # elif REF_PTS[0][0] < REF_PTS[1][0] and REF_PTS[0][1] > REF_PTS[1][1]:
        #     lb = REF_PTS[0]
        #     rt = REF_PTS[1]
        #     REF_PTS[0] = (lb[0], rt[1])
        #     REF_PTS[1] = (rt[0], lb[1])
        REF_PTS = list()
        cv2.destroyAllWindows()
        return frame, None  # result


if __name__ == "__main__":
    houses_folder = Path('sydney-house/rent_crawler/goodhouses')
    houses = list()
    for house_folder in houses_folder.iterdir():
        for house in house_folder.iterdir():
            if 'floorplan_label' in str(house) and house.suffix == '.png':
                houses.append(house)

    annotator = ManualAnnotator(houses, Path('processed_houses'))
    annotator.run()
