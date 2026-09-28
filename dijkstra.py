import heapq
import math

from data import Graph


def dijkstra(graph: Graph, start: int, goal: int):
    """
    Dijkstra's shortest-path algorithm.

    Edge cost is travel time in seconds.

    Returns:
        path: list of vertex IDs from start to goal
        cost: total travel time in seconds
    """

    distances = {
        start: 0.0
    }

    came_from = {}

    priority_queue = [
        (0.0, start)
    ]

    while priority_queue:

        current_distance, current = heapq.heappop(
            priority_queue
        )

        # Ignore an outdated queue entry.
        if current_distance > distances.get(
            current,
            math.inf
        ):
            continue

        if current == goal:
            return reconstruct_path(
                came_from,
                current
            ), current_distance

        for edge in graph.neighbors(current):

            if edge.blocked:
                continue

            neighbor = edge.target

            new_distance = (
                current_distance
                + edge.travel_time
            )

            if new_distance < distances.get(
                neighbor,
                math.inf
            ):
                distances[neighbor] = new_distance
                came_from[neighbor] = current

                heapq.heappush(
                    priority_queue,
                    (new_distance, neighbor)
                )

    return None, math.inf


def reconstruct_path(came_from, current):
    path = [current]

    while current in came_from:
        current = came_from[current]
        path.append(current)

    path.reverse()

    return path