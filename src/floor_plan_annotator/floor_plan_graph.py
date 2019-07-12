class Coordinate:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y


class Node:
    def __init__(self, x: float, y: float, length: float):
        self.coordinate = Coordinate(x, y)
        self.x = x
        self.y = y
        self.length = length


class FloorPlanGraph:
    def __init__(self):
        pass
