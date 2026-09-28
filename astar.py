import heapq
import math
from data import Graph, haversine_m, road_midpoint


def heuristic(graph: Graph, node: int, goal: int) -> float:
    """
    Optimistic estimate of travel time from node to goal.

    h(n) = straight-line distance(n, goal) / maximum possible speed.

    Returns:
        Estimated travel time in seconds.
    """

    v_max = graph.v_max / 3.6  # km/h -> m/s

    if v_max <= 0:
        return math.inf

    current = graph.vertices[node]
    target = graph.vertices[goal]

    distance = haversine_m(
        current.y, current.x,
        target.y, target.x
    )

    return distance / v_max

def heuristic_birdfly(graph: Graph, node: int, goal: int, samples = 10, local_vmax = False) -> float:
    current = graph.vertices[node]
    target = graph.vertices[goal]

    distance = haversine_m(
        current.y, current.x,
        target.y, target.x
    )

    if distance == 0:
        return 0.0

    # --------------------------------------------------
    # 2. Generate sample points along the bird-flight line
    # --------------------------------------------------

    reference_edges = []

    for i in range(1, samples + 1):

        # Fraction along the line:
        # 0 < t <= 1
        t = i / samples

        lat = current.y + t * (target.y - current.y)
        lon = current.x + t * (target.x - current.x)

        # --------------------------------------------------
        # 3. Find nearest road midpoint
        # --------------------------------------------------

        nearest_edge = None
        nearest_distance = math.inf

        for edges in graph.adj.values():
            for edge in edges.values():

                midpoint_lat, midpoint_lon = road_midpoint(
                    edge.geometry
                )

                d = haversine_m(
                    lat,
                    lon,
                    midpoint_lat,
                    midpoint_lon
                )

                if d < nearest_distance:
                    nearest_distance = d
                    nearest_edge = edge

        if nearest_edge is not None:
            reference_edges.append(nearest_edge)

    # No roads found
    if not reference_edges:
        return math.inf
    
    # --------------------------------------------------
    # 4. Average density and weather
    # --------------------------------------------------

    avg_density = (
        sum(edge.density for edge in reference_edges)
        / len(reference_edges)
    )

    avg_weather = (
        sum(edge.weather_factor for edge in reference_edges)
        / len(reference_edges)
    )

    # --------------------------------------------------
    # 5. Determine v_max
    # --------------------------------------------------

    if local_vmax:
        # Maximum speed limit among the sampled roads
        v_max = max(
            edge.speed_limit
            for edge in reference_edges
        )

    else:
        # Maximum speed limit in the entire graph
        v_max = graph.v_max

    v_max /= 3.6

    estimated_speed = (
        v_max
        * (1 - avg_density)
        * avg_weather
    )

    if estimated_speed <= 0:
        return math.inf

    return distance / estimated_speed

def astar(graph: Graph, start: int, goal: int, heuristic_fn = heuristic):
    """
    A* search using travel time as the edge cost.

    heuristic_fn must have the form:

        heuristic_fn(graph, node, goal) -> float

    Returns:
        path: list of vertex IDs from start to goal
        cost: total travel time in seconds
    """

    open_set = []

    g_score = {
        start: 0.0
    }

    came_from = {}

    h_start = heuristic_fn(
        graph,
        start,
        goal
    )

    heapq.heappush(
        open_set,
        (h_start, start)
    )

    while open_set:

        _, current = heapq.heappop(open_set)

        if current == goal:
            return (
                reconstruct_path(came_from, current),
                g_score[current]
            )

        for edge in graph.neighbors(current):

            if edge.blocked:
                continue

            neighbor = edge.target

            tentative_g = (
                g_score[current]
                + edge.travel_time
            )

            if tentative_g < g_score.get(
                neighbor,
                math.inf
            ):
                came_from[neighbor] = current

                g_score[neighbor] = tentative_g

                h = heuristic_fn(
                    graph,
                    neighbor,
                    goal
                )

                f_score = tentative_g + h

                heapq.heappush(
                    open_set,
                    (f_score, neighbor)
                )

    return None, math.inf

def reconstruct_path(came_from, current):
    path = [current]

    while current in came_from:
        current = came_from[current]
        path.append(current)

    path.reverse()

    return path