from typing import List, Tuple
from python_shape_grammars.floor_plan_elements import Node, Edge, RoomNode, \
    CornerNode, WallNode, Rectangle, Window, Door, Staircase
from python_shape_grammars.components import EdgeType, EdgeDirection, \
    RoomType, FloorPlanStatus
from python_shape_grammars.floor_plan import FloorPlan
from python_shape_grammars.vector import Vector
from python_shape_grammars.room import Room


def add_room(
        floor_plan: FloorPlan, corners: List[Node],
        room_type: RoomType, name,
        midpoint: Tuple[float, float]) -> Tuple[FloorPlan,
                                                Tuple[Node, Node,
                                                      Node, Node]]:
    node_a = corners[0]
    node_b = corners[1]
    node_c = corners[2]
    node_d = corners[3]
    room_node = RoomNode(
        Vector(midpoint[0], midpoint[1]), room_type=RoomType.staircase)
    edge_a = Edge(edge_type=EdgeType.wall, node_a=node_a, node_b=node_b)
    edge_b = Edge(edge_type=EdgeType.wall, node_a=node_b, node_b=node_c)
    edge_c = Edge(edge_type=EdgeType.wall, node_a=node_c, node_b=node_d)
    edge_d = Edge(edge_type=EdgeType.wall, node_a=node_d, node_b=node_a)
    edge_la = Edge(edge_type=EdgeType.wall, node_a=node_a, node_b=room_node)
    edge_lb = Edge(edge_type=EdgeType.wall, node_a=node_b, node_b=room_node)
    edge_lc = Edge(edge_type=EdgeType.wall, node_a=node_c, node_b=room_node)
    edge_ld = Edge(edge_type=EdgeType.wall, node_a=node_d, node_b=room_node)
    node_a.add_edge(edge_a)
    node_a.add_edge(edge_d)
    node_a.add_edge(edge_la)
    node_b.add_edge(edge_a)
    node_b.add_edge(edge_b)
    node_b.add_edge(edge_lb)
    node_c.add_edge(edge_b)
    node_c.add_edge(edge_c)
    node_c.add_edge(edge_lc)
    node_d.add_edge(edge_c)
    node_d.add_edge(edge_d)
    node_d.add_edge(edge_ld)
    room = Room(corners=[node_a, node_b, node_c, node_d],
                name=name,
                room_node=room_node)
    floor_plan.add_node(node_a)
    floor_plan.add_node(node_b)
    floor_plan.add_node(node_c)
    floor_plan.add_node(node_d)
    floor_plan.add_node(room_node)
    floor_plan.add_edge(edge_a)
    floor_plan.add_edge(edge_b)
    floor_plan.add_edge(edge_c)
    floor_plan.add_edge(edge_d)
    floor_plan.add_edge(edge_la)
    floor_plan.add_edge(edge_lb)
    floor_plan.add_edge(edge_lc)
    floor_plan.add_edge(edge_ld)
    floor_plan.add_room(room)
    return floor_plan, (node_a, node_b, node_c, node_d)


def add_adj_room(
        floor_plan: FloorPlan, corners: List[Node],
        room_type: RoomType, name,
        midpoint: Tuple[float, float]) -> Tuple[FloorPlan,
                                                Tuple[Node, Node,
                                                      Node, Node]]:
    node_a = corners[0]
    node_b = corners[1]
    node_c = corners[2]
    node_d = corners[3]
    room_node = RoomNode(
        Vector(midpoint[0], midpoint[1]), room_type=RoomType.staircase)
    edge_a = Edge(edge_type=EdgeType.wall, node_a=node_a, node_b=node_b)
    edge_b = Edge(edge_type=EdgeType.wall, node_a=node_b, node_b=node_c)
    edge_c = Edge(edge_type=EdgeType.wall, node_a=node_c, node_b=node_d)
    edge_d = Edge(edge_type=EdgeType.wall, node_a=node_d, node_b=node_a)
    edge_la = Edge(edge_type=EdgeType.wall, node_a=node_a, node_b=room_node)
    edge_lb = Edge(edge_type=EdgeType.wall, node_a=node_b, node_b=room_node)
    edge_lc = Edge(edge_type=EdgeType.wall, node_a=node_c, node_b=room_node)
    edge_ld = Edge(edge_type=EdgeType.wall, node_a=node_d, node_b=room_node)
    node_a.add_edge(edge_a)
    node_a.add_edge(edge_d)
    node_a.add_edge(edge_la)
    node_b.add_edge(edge_a)
    node_b.add_edge(edge_b)
    node_b.add_edge(edge_lb)
    node_c.add_edge(edge_b)
    node_c.add_edge(edge_c)
    node_c.add_edge(edge_lc)
    node_d.add_edge(edge_c)
    node_d.add_edge(edge_d)
    node_d.add_edge(edge_ld)
    room = Room(corners=[node_a, node_b, node_c, node_d],
                name=name,
                room_node=room_node)
    floor_plan.add_node(node_a)
    floor_plan.add_node(node_b)
    floor_plan.add_node(node_c)
    floor_plan.add_node(node_d)
    floor_plan.add_node(room_node)
    floor_plan.add_edge(edge_a)
    floor_plan.add_edge(edge_b)
    floor_plan.add_edge(edge_c)
    floor_plan.add_edge(edge_d)
    floor_plan.add_edge(edge_la)
    floor_plan.add_edge(edge_lb)
    floor_plan.add_edge(edge_lc)
    floor_plan.add_edge(edge_ld)
    floor_plan.add_room(room)
    return floor_plan, (node_a, node_b, node_c, node_d)


def add_wall(floor_plan: FloorPlan,
             node_a: Node,
             node_b: Node) -> Tuple[FloorPlan, Tuple[Node, Node]]:
    edge = Edge(edge_type=EdgeType.wall,
                node_a=node_a, node_b=node_b)
    node_a.add_edge(edge)
    node_b.add_edge(edge)
    floor_plan.add_node(node_a)
    floor_plan.add_node(node_b)
    floor_plan.add_edge(edge)
    return floor_plan, (node_a, node_b)


def house_32_floor_0():
    floor_plan = FloorPlan('house_32_floor_0', status=FloorPlanStatus.start)
    # origin = (6, 7)
    # footprint
    floor_plan, (node_a, node_b) = add_wall(floor_plan,
                                            node_a=Node(Vector(0, 0)),
                                            node_b=Node(Vector(58, 0)))
    floor_plan, (node_a, node_b) = add_wall(floor_plan,
                                            node_a=node_b,
                                            node_b=Node(Vector(386, 0)))
    floor_plan, (node_a, node_b) = add_wall(floor_plan,
                                            node_a=node_b,
                                            node_b=Node(Vector(386, -183)))
    floor_plan, (node_a, node_b) = add_wall(floor_plan,
                                            node_a=node_b,
                                            node_b=Node(Vector(184, -183)))
    floor_plan, (node_a, node_b) = add_wall(floor_plan,
                                            node_a=node_b,
                                            node_b=Node(Vector(184, -165)))
    floor_plan, (node_a, node_b) = add_wall(floor_plan,
                                            node_a=node_b,
                                            node_b=Node(Vector(127, -165)))
    floor_plan, (node_a, node_b) = add_wall(floor_plan,
                                            node_a=node_b,
                                            node_b=Node(Vector(127, -183)))
    floor_plan, (node_a, node_b) = add_wall(floor_plan,
                                            node_a=node_b,
                                            node_b=Node(Vector(14, -183)))
    floor_plan, (node_a, node_b) = add_wall(floor_plan,
                                            node_a=node_b,
                                            node_b=Node(Vector(14, -77)))
    floor_plan, (node_a, node_b) = add_wall(floor_plan,
                                            node_a=node_b,
                                            node_b=Node(Vector(56, -77)))
    floor_plan, (node_a, node_b) = add_wall(floor_plan,
                                            node_a=node_b,
                                            node_b=Node(Vector(58, 0)))

    floor_plan, (node_a, node_b, node_c, node_d) = add_room(
        floor_plan,
        corners=[Node(Vector(134, 0)), Node(Vector(201, 0)),
                 Node(Vector(201, -77)), Node(Vector(134, -77))],
        room_type=RoomType.staircase,
        name='stairs', midpoint=(167.5, -38.5))
    floor_plan, (node_a, node_b, node_c, node_d) = add_room(
        floor_plan,
        corners=[node_b, Node(Vector(254, 0)),
                 Node(Vector(254, -77)), node_c],
        room_type=RoomType.bath,
        name='stairs', midpoint=(227.5, -38.5))
    floor_plan, (node_a, node_b, node_c, node_d) = add_room(
        floor_plan,
        corners=[node_b, Node(Vector(341, 0)),
                 Node(Vector(341, -77)), node_c],
        room_type=RoomType.bed,
        name='stairs', midpoint=(297.5, -38.5))


if __name__ == "__main__":
    house_32_floor_0()
