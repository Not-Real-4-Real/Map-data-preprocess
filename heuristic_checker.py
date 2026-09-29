import heapq
import math

from data import Graph


def reverse_dijkstra(graph: Graph, goal: int):
    """
    Compute the true shortest travel-time cost from every vertex to the goal.

    Dijkstra runs from the goal on the reverse graph (radj).

    Returns:
        distances:
            node -> minimum travel time from node to goal
    """

    distances = {
        goal: 0.0
    }

    priority_queue = [
        (0.0, goal)
    ]

    while priority_queue:

        current_distance, current = heapq.heappop(
            priority_queue
        )

        # Ignore outdated queue entries.
        if current_distance > distances.get(
            current,
            math.inf
        ):
            continue

        # Original:
        #
        #     predecessor -----> current
        #
        # Reverse search:
        #
        #     current -----> predecessor
        #
        for edge in graph.predecessors(current):

            if edge.blocked:
                continue

            predecessor = edge.source

            new_distance = (
                current_distance
                + edge.travel_time
            )

            if new_distance < distances.get(
                predecessor,
                math.inf
            ):
                distances[predecessor] = new_distance

                heapq.heappush(
                    priority_queue,
                    (new_distance, predecessor)
                )

    return distances

def check_admissibility(
    graph: Graph,
    goal: int,
    heuristic_fn,
    tolerance=1e-9
):
    """
    Check whether a heuristic is admissible.

    Admissibility requires:

        h(n) <= h*(n)

    where h*(n) is the true shortest travel time from n to the goal.

    Returns:
        violations: list of dictionaries describing every violating node.
    """

    true_costs = reverse_dijkstra(
        graph,
        goal
    )

    violations = []

    for node, true_cost in true_costs.items():

        heuristic_cost = heuristic_fn(
            graph,
            node,
            goal
        )

        if heuristic_cost > true_cost + tolerance:

            violations.append({
                "node": node,
                "heuristic": heuristic_cost,
                "true_cost": true_cost,
                "overestimate": (
                    heuristic_cost - true_cost
                )
            })

    return violations

def check_consistency(
    graph: Graph,
    goal: int,
    heuristic_fn,
    tolerance=1e-9
):
    """
    Check whether a heuristic is consistent.

    For every usable edge u -> v:

        h(u) <= cost(u,v) + h(v)

    Returns:
        violations: list of dictionaries describing every violating edge.
    """

    violations = []

    for u in graph.adj:

        h_u = heuristic_fn(
            graph,
            u,
            goal
        )

        if not math.isfinite(h_u):
            continue

        for edge in graph.neighbors(u):

            if edge.blocked:
                continue

            v = edge.target

            h_v = heuristic_fn(
                graph,
                v,
                goal
            )

            if not math.isfinite(h_v):
                continue

            rhs = edge.travel_time + h_v

            if h_u > rhs + tolerance:

                violations.append({
                    "source": u,
                    "target": v,
                    "heuristic_source": h_u,
                    "edge_cost": edge.travel_time,
                    "heuristic_target": h_v,
                    "rhs": rhs,
                    "violation": h_u - rhs
                })

    return violations

def report_heuristic(
    graph: Graph,
    goal: int,
    heuristic_fn,
    name: str
):
    """
    Run and print admissibility and consistency checks.
    """

    print(f"\n========== {name} ==========")

    admissibility_violations = check_admissibility(
        graph,
        goal,
        heuristic_fn
    )

    consistency_violations = check_consistency(
        graph,
        goal,
        heuristic_fn
    )

    if not admissibility_violations:
        print("Admissibility: PASS")
    else:
        print(
            f"Admissibility: FAIL "
            f"({len(admissibility_violations)} violations)"
        )

        for violation in admissibility_violations[:5]:
            print(
                f"  node={violation['node']} | "
                f"h={violation['heuristic']:.4f} | "
                f"true={violation['true_cost']:.4f} | "
                f"over={violation['overestimate']:.4f}"
            )

    if not consistency_violations:
        print("Consistency: PASS")
    else:
        print(
            f"Consistency: FAIL "
            f"({len(consistency_violations)} violations)"
        )

        for violation in consistency_violations[:5]:
            print(
                f"  {violation['source']} -> "
                f"{violation['target']} | "
                f"h(u)={violation['heuristic_source']:.4f} | "
                f"c={violation['edge_cost']:.4f} | "
                f"h(v)={violation['heuristic_target']:.4f}"
            )

    return (
        admissibility_violations,
        consistency_violations
    )