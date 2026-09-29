import time
import math

from data import (
    parse_maxspeed,
    load_graph,
    ScenarioConfig,
    ScenarioGenerator,
    Hotspot,
    RainCell,
    road_midpoint,
    _roads,
    haversine_m
)

from astar import heuristic, heuristic_birdfly, heuristic_birdfly_v2, astar
from dijkstra import dijkstra
from heuristic_checker import report_heuristic

def generate_test_data():
    large_bbox = (
        106.688,
        10.765,
        106.708,
        10.785
    )

    g = load_graph(large_bbox)

    cfg = ScenarioConfig(
        base_density=0.10,
        max_density=0.90,
        density_deviation=0.05,
        block_probability=0.01,

        hotspots=[
            Hotspot(
                lat=10.7705,
                lon=106.6935,
                intensity=0.75,
                radius=350
            ),
            Hotspot(
                lat=10.7765,
                lon=106.7015,
                intensity=0.90,
                radius=450
            ),
            Hotspot(
                lat=10.7800,
                lon=106.6960,
                intensity=0.65,
                radius=300
            ),
            Hotspot(
                lat=10.7730,
                lon=106.7040,
                intensity=0.80,
                radius=350
            )
        ],

        global_weather_factor=1.0,

        rain_cells=[
            RainCell(
                min_lat=10.768,
                max_lat=10.774,
                min_lon=106.692,
                max_lon=106.698,
                weather_factor=0.65
            ),
            RainCell(
                min_lat=10.776,
                max_lat=10.783,
                min_lon=106.700,
                max_lon=106.706,
                weather_factor=0.55
            )
        ]
    )

    ScenarioGenerator(
        seed=424545,
        config=cfg
    ).apply(g)

    vertices = list(g.vertices.keys())
    start = vertices[0]

    # Find all reachable vertices.
    reachable = set()
    stack = [start]

    while stack:
        current = stack.pop()

        if current in reachable:
            continue

        reachable.add(current)

        for edge in g.neighbors(current):
            if not edge.blocked:
                stack.append(edge.target)

    if len(reachable) <= 1:
        raise RuntimeError(
            "No reachable destination exists."
        )

    # Choose geographically farthest reachable vertex.
    start_vertex = g.vertices[start]

    goal = max(
        reachable - {start},
        key=lambda node: haversine_m(
            start_vertex.y,
            start_vertex.x,
            g.vertices[node].y,
            g.vertices[node].x
        )
    )

    return g, start, goal, cfg

def run_test(g, start, goal):

    tests = [
        ("Basic A*", heuristic),
        ("Bird-flight A*", heuristic_birdfly),
        ("Bird-flight V2", heuristic_birdfly_v2),
    ]

    results = {}

    for name, h in tests:

        start_time = time.perf_counter()

        path, cost = astar(
            g,
            start,
            goal,
            h
        )

        elapsed = time.perf_counter() - start_time

        results[name] = {
            "path": path,
            "cost": cost,
            "time": elapsed,
            "heuristic": h
        }

    # Dijkstra separately because it has no heuristic.
    start_time = time.perf_counter()

    path, cost = dijkstra(
        g,
        start,
        goal
    )

    elapsed = time.perf_counter() - start_time

    results["Dijkstra"] = {
        "path": path,
        "cost": cost,
        "time": elapsed,
        "heuristic": None
    }

    return results

def report_results(g, start, goal, results):

    print("\n========== RESULTS ==========")

    for name, result in results.items():

        path = result["path"]
        cost = result["cost"]
        elapsed = result["time"]

        print(f"\n{name}")

        if path is None:
            print("  No path found.")
            continue

        print(f"  Path found: {len(path)} vertices")
        print(f"  Cost: {cost:.2f} seconds")
        print(f"  Cost: {cost / 60:.2f} minutes")
        print(f"  Search time: {elapsed * 1000:.3f} ms")

        print("  Path:")
        print("   ", " -> ".join(map(str, path)))

    # Dijkstra is our reference.
    dijkstra_result = results["Dijkstra"]

    assert dijkstra_result["path"] is not None

    dijkstra_cost = dijkstra_result["cost"]

    print("\n========== COMPARISON ==========")

    for name, result in results.items():

        if name == "Dijkstra":
            continue

        assert result["path"] is not None

        difference = abs(
            result["cost"] - dijkstra_cost
        )

        print(
            f"{name} difference: "
            f"{difference:.10f} seconds"
        )

        assert math.isclose(
            result["cost"],
            dijkstra_cost,
            rel_tol=1e-9,
            abs_tol=1e-9
        )

    # Verify every path.
    for name, result in results.items():

        path = result["path"]
        cost = result["cost"]

        assert path[0] == start
        assert path[-1] == goal
        assert math.isfinite(cost)

        for u, v in zip(path, path[1:]):

            edge = g.get_edge(u, v)

            assert edge is not None
            assert not edge.blocked

    # Heuristic checks.
    for name, result in results.items():

        h = result["heuristic"]

        if h is None:
            continue

        report_heuristic(
            g,
            goal,
            h,
            name.upper()
        )

    print("\nAll A* and Dijkstra tests passed.")

def test3():
    # ---------- 1. Load graph ----------
    small_bbox = (106.694, 10.769, 106.702, 10.777)
    g = load_graph(small_bbox)

    print("\n========== A* TEST ==========")

    print(
        f"{len(g.vertices)} vertices, "
        f"{sum(len(t) for t in g.adj.values())} directed edges"
    )

    print(f"v_max = {g.v_max:.2f} km/h")

    # ---------- 2. Generate traffic scenario ----------
    cfg = ScenarioConfig(
        base_density=0.1,
        max_density=0.9,
        density_deviation=0.05,
        block_probability=0.00,

        hotspots=[
            Hotspot(
                lat=10.773,
                lon=106.698,
                intensity=0.75,
                radius=400
            )
        ],

        global_weather_factor=1.0,

        rain_cells=[
            RainCell(
                min_lat=10.771,
                max_lat=10.775,
                min_lon=106.696,
                max_lon=106.700,
                weather_factor=0.6
            )
        ]
    )

    ScenarioGenerator(
        seed=424545,
        config=cfg
    ).apply(g)

    # ---------- 3. Pick start and goal ----------
    vertices = list(g.vertices.keys())

    start = vertices[0]
    goal = vertices[-1]

    print(f"\nStart vertex: {start}")
    print(f"Goal vertex:  {goal}")

    # Check whether start and goal are connected
    print(
        f"Outgoing edges from start: "
        f"{len(g.adj[start])}"
    )

    print(
        f"Incoming edges to goal: "
        f"{len(g.radj[goal])}"
    )

    # Show the start's outgoing edges
    print("\nStart outgoing edges:")

    for edge in g.neighbors(start):
        print(
            f"  {edge.source} -> {edge.target} | "
            f"blocked={edge.blocked} | "
            f"time={edge.travel_time:.2f}s"
        )

    # ---------- 4. Find a guaranteed reachable goal ----------
    #
    # For testing A*, first choose a goal that is reachable
    # from the start through currently unblocked directed edges.

    reachable = set()
    stack = [start]

    while stack:
        current = stack.pop()

        if current in reachable:
            continue

        reachable.add(current)

        for edge in g.neighbors(current):
            if not edge.blocked:
                stack.append(edge.target)

    print(
        f"\nReachable vertices from start: "
        f"{len(reachable)} / {len(g.vertices)}"
    )

    if len(reachable) <= 1:
        print("No reachable destination exists.")
        return

    # Choose a reachable destination different from start.
    goal = next(
        v for v in reversed(vertices)
        if v in reachable and v != start
    )

    print(f"Selected reachable goal: {goal}")

    # ---------- 5. Test heuristics ----------

    print("\n========== HEURISTIC TEST ==========")

    h_basic = heuristic(
        g,
        start,
        goal
    )

    h_birdfly = heuristic_birdfly(
        g,
        start,
        goal,
        samples=10,
        local_vmax=False
    )

    print("\nBasic heuristic:")
    print(f"  h(start) = {h_basic:.2f} seconds")

    print("\nBird-flight heuristic:")
    print(f"  h(start) = {h_birdfly:.2f} seconds")

    # Both heuristics should be zero at the goal.
    h_basic_goal = heuristic(
        g,
        goal,
        goal
    )

    h_birdfly_goal = heuristic_birdfly(
        g,
        goal,
        goal,
        samples=10,
        local_vmax=True
    )

    print("\nHeuristic at goal:")
    print(f"  Basic:     {h_basic_goal:.2f} seconds")
    print(f"  Bird-flight: {h_birdfly_goal:.2f} seconds")

    assert math.isclose(
        h_basic_goal,
        0.0
    )

    assert math.isclose(
        h_birdfly_goal,
        0.0
    )


    # ---------- 6. Run A* with basic heuristic ----------

    start_time = time.perf_counter()

    astar_basic_path, astar_basic_cost = astar(
        g,
        start,
        goal,
        heuristic
    )

    astar_basic_elapsed = (
        time.perf_counter() - start_time
    )


    # ---------- 7. Run A* with bird-flight heuristic ----------

    start_time = time.perf_counter()

    astar_birdfly_path, astar_birdfly_cost = astar(
        g,
        start,
        goal,
        heuristic_birdfly
    )

    astar_birdfly_elapsed = (
        time.perf_counter() - start_time
    )


    # ---------- 8. Run Dijkstra ----------

    start_time = time.perf_counter()

    dijkstra_path, dijkstra_cost = dijkstra(
        g,
        start,
        goal
    )

    dijkstra_elapsed = (
        time.perf_counter() - start_time
    )


    # ---------- 9. Print A* basic result ----------

    print("\n========== A* BASIC HEURISTIC ==========")

    if astar_basic_path is None:
        print("  No path found.")
    else:
        print(
            f"  Path found: "
            f"{len(astar_basic_path)} vertices"
        )

        print(
            f"  Total travel time: "
            f"{astar_basic_cost:.2f} seconds"
        )

        print(
            f"  Total travel time: "
            f"{astar_basic_cost / 60:.2f} minutes"
        )

        print(
            f"  Search time: "
            f"{astar_basic_elapsed * 1000:.3f} ms"
        )

        print("\n  Path:")
        print(
            "   ",
            " -> ".join(
                map(str, astar_basic_path)
            )
        )


    # ---------- 10. Print A* bird-flight result ----------

    print("\n========== A* BIRD-FLIGHT HEURISTIC ==========")

    if astar_birdfly_path is None:
        print("  No path found.")
    else:
        print(
            f"  Path found: "
            f"{len(astar_birdfly_path)} vertices"
        )

        print(
            f"  Total travel time: "
            f"{astar_birdfly_cost:.2f} seconds"
        )

        print(
            f"  Total travel time: "
            f"{astar_birdfly_cost / 60:.2f} minutes"
        )

        print(
            f"  Search time: "
            f"{astar_birdfly_elapsed * 1000:.3f} ms"
        )

        print("\n  Path:")
        print(
            "   ",
            " -> ".join(
                map(str, astar_birdfly_path)
            )
        )


    # ---------- 11. Print Dijkstra result ----------

    print("\n========== DIJKSTRA ==========")

    if dijkstra_path is None:
        print("  No path found.")
    else:
        print(
            f"  Path found: "
            f"{len(dijkstra_path)} vertices"
        )

        print(
            f"  Total travel time: "
            f"{dijkstra_cost:.2f} seconds"
        )

        print(
            f"  Total travel time: "
            f"{dijkstra_cost / 60:.2f} minutes"
        )

        print(
            f"  Search time: "
            f"{dijkstra_elapsed * 1000:.3f} ms"
        )

        print("\n  Path:")
        print(
            "   ",
            " -> ".join(
                map(str, dijkstra_path)
            )
        )


    # ---------- 12. Compare algorithms ----------

    print("\n========== COMPARISON ==========")

    assert astar_basic_path is not None
    assert astar_birdfly_path is not None
    assert dijkstra_path is not None


    # All algorithms should currently find
    # the same optimal travel cost.

    assert math.isclose(
        astar_basic_cost,
        dijkstra_cost,
        rel_tol=1e-9,
        abs_tol=1e-9
    )

    assert math.isclose(
        astar_birdfly_cost,
        dijkstra_cost,
        rel_tol=1e-9,
        abs_tol=1e-9
    )


    print(
        f"Basic A* cost:       "
        f"{astar_basic_cost:.2f} seconds"
    )

    print(
        f"Bird-flight A* cost: "
        f"{astar_birdfly_cost:.2f} seconds"
    )

    print(
        f"Dijkstra cost:       "
        f"{dijkstra_cost:.2f} seconds"
    )


    print(
        f"\nBasic A* difference: "
        f"{abs(astar_basic_cost - dijkstra_cost):.10f} seconds"
    )

    print(
        f"Bird-flight difference: "
        f"{abs(astar_birdfly_cost - dijkstra_cost):.10f} seconds"
    )


    print(
        f"\nBasic A* search time: "
        f"{astar_basic_elapsed * 1000:.3f} ms"
    )

    print(
        f"Bird-flight A* search time: "
        f"{astar_birdfly_elapsed * 1000:.3f} ms"
    )

    print(
        f"Dijkstra search time: "
        f"{dijkstra_elapsed * 1000:.3f} ms"
    )


    # ---------- 13. Verify paths ----------

    for path, cost in [
        (astar_basic_path, astar_basic_cost),
        (astar_birdfly_path, astar_birdfly_cost),
        (dijkstra_path, dijkstra_cost)
    ]:

        assert path[0] == start
        assert path[-1] == goal
        assert math.isfinite(cost)

        for u, v in zip(path, path[1:]):

            edge = g.get_edge(u, v)

            assert edge is not None
            assert not edge.blocked


    print("\nAll A* and Dijkstra tests passed.")

    # ---------- 14. Verify heuristic attr ----------
    report_heuristic(
    g,
    goal,
    heuristic,
    "BASIC HEURISTIC"
    )

    
    report_heuristic(
    g,
    goal,
    heuristic_birdfly,
    "BIRD-FLIGHT HEURISTIC"
    )

def test4():
    g, start, goal, cfg = generate_test_data()

    results = run_test(
        g,
        start,
        goal
    )

    report_results(
        g,
        start,
        goal,
        results
    )

if __name__ == "__main__":
    test4()
