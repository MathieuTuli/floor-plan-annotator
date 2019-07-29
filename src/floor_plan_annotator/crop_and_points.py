# TODO: Make Frame class that has frame: np.ndarray and annotions: List[str] as
# @ members that can handle all the drawing on its own
from pathlib import Path
from typing import Tuple, List

import numpy as np
import cv2
import sys

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
        self.origin = None
        self.current_floor = None
        self.prev_points = None
        self.points = list()

    @staticmethod
    def record_click(event, x, y, flags, param):
        global REF_PTS
        global CLICK_BARRIER
        CLICK_BARRIER = False
        if event == cv2.EVENT_LBUTTONDOWN:
            REF_PTS = [(x, y)]
            CLICK_BARRIER = True
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

    def run(self,) -> None:
        for floor_plan in self.floor_plans:
            frame = cv2.imread(str(floor_plan))
            frame_copy = frame.copy()
            self.origin = None
            self.current_floor = None
            while True:
                cv2.destroyAllWindows()
                cv2.namedWindow("image")
                cv2.moveWindow("image", 500, 200)
                cv2.imshow("image", frame_copy)
                print('\n')
                print('What would you like do?\n' +
                      '[c] to crop\n' +
                      '[p] to read points\n'
                      '[q] to quit\n'
                      'Press [n] to move on')
                key = cv2.waitKey(0)
                key = key & 0xFF
                if key == ord('c'):
                    frame_copy = self.crop(frame, str(floor_plan))
                elif key == ord('p'):
                    self.print_points(frame_copy)
                elif key == ord('q'):
                    print('\n')
                    print("Are you sure you want to quit [y]/[n]")
                    while True:
                        q = cv2.waitKey(0)
                        if q & 0xFF == ord('y'):
                            sys.exit(0)
                        elif q & 0xFF == ord('n'):
                            break
                elif key == ord('n'):
                    print('\n')
                    print("Are you sure you want to move on [y]/[n]")
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

    def print_points(self, frame: np.ndarray) -> Tuple[np.ndarray, None]:
        print('\n')
        print("POINTS")
        cv2.destroyAllWindows()
        cv2.namedWindow("clickable_image")
        cv2.moveWindow("clickable_image", 500, 200)
        cv2.setMouseCallback("clickable_image",
                             ManualAnnotator.record_click)
        old_frame = frame.copy()
        printing_points = False
        while True:
            global REF_PTS
            cv2.imshow('clickable_image', frame)
            print("Click or set origin (TL) [o]")
            print("[p] to print points")
            print("[q] to quit")
            if not printing_points:
                key = cv2.waitKey(0)
            if key & 0xFF == ord('o'):
                if len(REF_PTS) == 2:
                    print('\n')
                    print(f"Origin is {REF_PTS[0]}")
                    self.origin = REF_PTS[0]
                    self.points.append((0, 0))
            elif key & 0xFF == ord('p'):
                if len(REF_PTS) == 2:
                    cv2.circle(frame, REF_PTS[0],
                               radius=4, thickness=-1, color=(255, 0, 0))
                    current_point = (REF_PTS[0][0] - self.origin[0],
                                     REF_PTS[0][1] - self.origin[1])
                    if self.origin is not None:
                        for point in self.points:
                            min_y = 0
                            min_x = 0
                            diff_min_x = 1000000
                            diff_min_y = 1000000
                            if abs(current_point[0] - point[0]) < diff_min_x:
                                diff_min_x = abs(current_point[0] - point[0])
                                min_x = point[0]
                            if abs(current_point[1] - point[1]) < diff_min_y:
                                diff_min_y = abs(current_point[1] - point[1])
                                min_y = point[1]
                        if diff_min_x < diff_min_y:
                            current_point = (min_x, current_point[1])
                        elif diff_min_x > diff_min_y:
                            current_point = (current_point[0], min_y)
                        else:
                            current_point = (min_x, min_y)
                        self.points.append(current_point)
                        print(f"X: {current_point[0]}")
                        print(f"Y: {current_point[1]}")
                REF_PTS = list()
            elif key & 0xFF == ord('q'):
                break
        REF_PTS = list()
        cv2.destroyAllWindows()
        return frame, None  # result

    def get_tl_br(self, ref_pts):
        if ref_pts[0][0] > ref_pts[1][0] and ref_pts[0][1] < ref_pts[1][1]:
            rt = ref_pts[0]
            lb = ref_pts[1]
            ref_pts[0] = (lb[0], rt[1])
            ref_pts[1] = (rt[0], lb[1])
        # drawn rb to lt
        elif ref_pts[0][0] > ref_pts[1][0] and ref_pts[0][1] > ref_pts[1][1]:
            temp = ref_pts[0]
            ref_pts[0] = ref_pts[1]
            ref_pts[1] = temp
        # drawn lb to rt
        elif ref_pts[0][0] < ref_pts[1][0] and ref_pts[0][1] > ref_pts[1][1]:
            lb = ref_pts[0]
            rt = ref_pts[1]
            ref_pts[0] = (lb[0], rt[1])
            ref_pts[1] = (rt[0], lb[1])

        return ref_pts

    def crop(self, frame: np.ndarray, save_to) -> np.ndarray:
        print('\n')
        print("CROPING")
        cv2.destroyAllWindows()
        cv2.namedWindow("clickable_image")
        cv2.moveWindow("clickable_image", 500, 200)
        cv2.setMouseCallback("clickable_image",
                             ManualAnnotator.record_click)
        old_frame = frame.copy()
        while True:
            global REF_PTS
            if len(REF_PTS) == 2:
                REF_PTS = self.get_tl_br(REF_PTS)
                cv2.imshow('clickable_image',
                           frame[REF_PTS[0][1]:REF_PTS[1][1],
                                 REF_PTS[0][0]:REF_PTS[1][0]])
                print('\n')
                print("[y] to confirm")
                print("[*] to decline")
                print("[q] to quit")
                key = cv2.waitKey(0)
                if key & 0xFF == ord('y'):
                    print("confirmed")
                    frame = frame[REF_PTS[0][1]:REF_PTS[1][1],
                                  REF_PTS[0][0]:REF_PTS[1][0]]
                    while True:
                        self.current_floor = input(
                            "Current floor num [-2, -1, 0, 1, 2, ...]: ")
                        try:
                            self.current_floor = int(self.current_floor)
                            break
                        except Exception:
                            continue
                    save_to = save_to.split('/')
                    house_num = save_to[3]
                    path = Path(self.save_to / house_num)
                    path.mkdir(parents=True,
                               exist_ok=True)
                    path = str(path / f'floor_{self.current_floor}.jpg')
                    cv2.imwrite(str(path), frame)
                    print('\n')
                    print(f"Saved to {str(path)}")
                    return frame
                elif key & 0xFF == ord('q'):
                    return old_frame
                else:
                    frame = old_frame.copy()
                REF_PTS = list()
            else:
                cv2.imshow('clickable_image', frame)
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    print('\n')
                    print("Quitting session")
                    return old_frame
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

    annotator = ManualAnnotator(houses, Path('processed_houses/floor-plans'))
    annotator.run()
