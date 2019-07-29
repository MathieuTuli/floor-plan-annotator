from python_shape_grammars.floor_plan_elements import Node, Edge, RoomNode, \
    CornerNode, WallNode, Rectangle, Window, Door, Staircase
from python_shape_grammars.components import EdgeType, EdgeDirection, \
    RoomType, FloorPlanStatus
from python_shape_grammars.floor_plan import FloorPlan
from python_shape_grammars.vector import Vector
from python_shape_grammars.room import Room


def house_32_floor_0():
    floor_plan = FloorPlan('house_32_floor_0', status=FloorPlanStatus.start)
    # origin = (6, 7)
    # footprint
    node_a = CornerNode(Vector(0, 0))
    node_b = CornerNode(Vector(58, 0))
    edge = Edge(edge_type=EdgeType.wall,
                node_a=node_a, node_b=node_b)
    node_a.add_edge(edge)
    node_b.add_edge(edge)
    floor_plan.add_node(node_a)
    floor_plan.add_node(node_b)
    floor_plan.add_edge(edge)

    node_a = CornerNode(Vector(58, 0))
    node_b = CornerNode(Vector(386, 0))
    edge = Edge(edge_type=EdgeType.wall,
                node_a=node_a, node_b=node_b)
    node_a.add_edge(edge)
    node_b.add_edge(edge)
    floor_plan.add_node(node_a)
    floor_plan.add_node(node_b)
    floor_plan.add_edge(edge)

    node_a = CornerNode(Vector(386, 0))
    node_b = CornerNode(Vector(386, 183))
    edge = Edge(edge_type=EdgeType.wall,
                node_a=node_a, node_b=node_b)
    node_a.add_edge(edge)
    node_b.add_edge(edge)
    floor_plan.add_node(node_a)
    floor_plan.add_node(node_b)
    floor_plan.add_edge(edge)

    node_a = CornerNode(Vector(386, 183))
    node_b = CornerNode(Vector(184, 183))
    edge = Edge(edge_type=EdgeType.wall,
                node_a=node_a, node_b=node_b)
    node_a.add_edge(edge)
    node_b.add_edge(edge)
    floor_plan.add_node(node_a)
    floor_plan.add_node(node_b)
    floor_plan.add_edge(edge)

    node_a = CornerNode(Vector(184, 183))
    node_b = CornerNode(Vector(184, 165))
    edge = Edge(edge_type=EdgeType.wall,
                node_a=node_a, node_b=node_b)
    node_a.add_edge(edge)
    node_b.add_edge(edge)
    floor_plan.add_node(node_a)
    floor_plan.add_node(node_b)
    floor_plan.add_edge(edge)

    node_a = CornerNode(Vector(184, 165))
    node_b = CornerNode(Vector(127, 165))
    edge = Edge(edge_type=EdgeType.wall,
                node_a=node_a, node_b=node_b)
    node_a.add_edge(edge)
    node_b.add_edge(edge)
    floor_plan.add_node(node_a)
    floor_plan.add_node(node_b)
    floor_plan.add_edge(edge)

    node_a = CornerNode(Vector(127, 165))
    node_b = CornerNode(Vector(127, 183))
    edge = Edge(edge_type=EdgeType.wall,
                node_a=node_a, node_b=node_b)
    node_a.add_edge(edge)
    node_b.add_edge(edge)
    floor_plan.add_node(node_a)
    floor_plan.add_node(node_b)
    floor_plan.add_edge(edge)

    node_a = CornerNode(Vector(127, 183))
    node_b = CornerNode(Vector(14, 183))
    edge = Edge(edge_type=EdgeType.wall,
                node_a=node_a, node_b=node_b)
    node_a.add_edge(edge)
    node_b.add_edge(edge)
    floor_plan.add_node(node_a)
    floor_plan.add_node(node_b)
    floor_plan.add_edge(edge)

    node_a = CornerNode(Vector(14, 183))
    node_b = CornerNode(Vector(14, 77))
    edge = Edge(edge_type=EdgeType.wall,
                node_a=node_a, node_b=node_b)
    node_a.add_edge(edge)
    node_b.add_edge(edge)
    floor_plan.add_node(node_a)
    floor_plan.add_node(node_b)
    floor_plan.add_edge(edge)

    node_a = CornerNode(Vector(14, 77))
    node_b = CornerNode(Vector(56, 77))
    edge = Edge(edge_type=EdgeType.wall,
                node_a=node_a, node_b=node_b)
    node_a.add_edge(edge)
    node_b.add_edge(edge)
    floor_plan.add_node(node_a)
    floor_plan.add_node(node_b)
    floor_plan.add_edge(edge)

    node_a = CornerNode(Vector(58, 77))
    node_b = CornerNode(Vector(58, 0))
    edge = Edge(edge_type=EdgeType.wall,
                node_a=node_a, node_b=node_b)
    node_a.add_edge(edge)
    node_b.add_edge(edge)
    floor_plan.add_node(node_a)
    floor_plan.add_node(node_b)
    floor_plan.add_edge(edge)

    # room
    node_a = Node(Vector(134, 0))
    node_b = Node(Vector(201, 0))
    node_c = Node(Vector(201, 77))
    node_d = Node(Vector(134, 77))
    room_node = RoomNode(Vector(167.5, 38.5), room_type=RoomType.staircase)
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
                name='stairs',
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

    # node_a = Node(Vector(134, 0))
    # node_b = Node(Vector(201, 0))
    # node_c = Node(Vector(201, 77))
    # node_d = Node(Vector(134, 77))
    # room_node = RoomNode(Vector(167.5, 38.5), room_type=RoomType.staircase)
    # edge_a = Edge(edge_type=EdgeType.wall, node_a=node_a, node_b=node_b)
    # edge_b = Edge(edge_type=EdgeType.wall, node_a=node_b, node_b=node_c)
    # edge_c = Edge(edge_type=EdgeType.wall, node_a=node_c, node_b=node_d)
    # edge_d = Edge(edge_type=EdgeType.wall, node_a=node_d, node_b=node_a)
    # edge_la = Edge(edge_type=EdgeType.wall, node_a=node_d, node_b=room_node)
    # edge_lb = Edge(edge_type=EdgeType.wall, node_a=node_d, node_b=room_node)
    # edge_lc = Edge(edge_type=EdgeType.wall, node_a=node_d, node_b=room_node)
    # edge_ld = Edge(edge_type=EdgeType.wall, node_a=node_d, node_b=room_node)
    # node_a.add_edge(edge_a)
    # node_a.add_edge(edge_d)
    # node_a.add_edge(edge_la)
    # node_b.add_edge(edge_a)
    # node_b.add_edge(edge_b)
    # node_b.add_edge(edge_lb)
    # node_c.add_edge(edge_b)
    # node_c.add_edge(edge_c)
    # node_c.add_edge(edge_lc)
    # node_d.add_edge(edge_c)
    # node_d.add_edge(edge_d)
    # node_d.add_edge(edge_ld)
    # room = Room(corners=[node_a, node_b, node_c, node_d],
    #             name='stairs',
    #             room_node=room_node)
    # floor_plan.add_node(node_a)
    # floor_plan.add_node(node_b)
    # floor_plan.add_node(node_c)
    # floor_plan.add_node(node_d)
    # floor_plan.add_node(room_node)
    # floor_plan.add_edge(edge_a)
    # floor_plan.add_edge(edge_b)
    # floor_plan.add_edge(edge_c)
    # floor_plan.add_edge(edge_d)
    # floor_plan.add_edge(edge_la)
    # floor_plan.add_edge(edge_lb)
    # floor_plan.add_edge(edge_lc)
    # floor_plan.add_edge(edge_ld)
    # floor_plan.add_room(room)


if __name__ == "__main__":
    house_32_floor_0()
