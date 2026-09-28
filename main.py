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
    _roads
)

from astar import heuristic, heuristic_birdfly, astar
from dijkstra import dijkstra

import folium
from folium.plugins import PolyLineTextPath
from branca.colormap import LinearColormap

def graph_to_geojson(graph):
    """
    Convert graph roads into a GeoJSON FeatureCollection.

    Each physical road is included once:
    - two-way road -> one feature
    - one-way road -> one feature per directed edge
    """

    features = []

    for edge, twin in _roads(graph):

        # GeoJSON expects coordinates as (longitude, latitude)
        coordinates = [
            (lon, lat)
            for lat, lon in edge.geometry
        ]

        features.append({
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": coordinates
            },
            "properties": {
                "density": edge.density,
                "blocked": edge.blocked,
                "oneway": edge.oneway,
                "name": edge.name,
                "speed_limit": edge.speed_limit,
                "current_speed": edge.current_speed,
                "travel_time": edge.travel_time
            }
        })

    return {
        "type": "FeatureCollection",
        "features": features
    }

def create_geojson_map(
    graph,
    output_file="test2_large_geojson.html"
):
    """
    Lightweight overview visualization.

    Layers:
        - Esri Streets
        - Road network GeoJSON

    No markers, arrows, tooltips, or individual
    Folium road objects.
    """

    geojson_data = graph_to_geojson(graph)

    # --------------------------------------------------------
    # Map center
    # --------------------------------------------------------

    cx = sum(
        v.x for v in graph.vertices.values()
    ) / len(graph.vertices)

    cy = sum(
        v.y for v in graph.vertices.values()
    ) / len(graph.vertices)

    # --------------------------------------------------------
    # Map
    # --------------------------------------------------------

    m = folium.Map(
        location=[cy, cx],
        zoom_start=13,
        tiles=None,
        prefer_canvas=True
    )

    # --------------------------------------------------------
    # Esri Streets
    # --------------------------------------------------------

    folium.TileLayer(
        tiles=(
            "https://server.arcgisonline.com/"
            "ArcGIS/rest/services/World_Street_Map/"
            "MapServer/tile/{z}/{y}/{x}"
        ),
        attr="Tiles © Esri",
        name="Esri Streets",
        max_zoom=19,
        overlay=False,
        control=True,
        show=True
    ).add_to(m)

    # --------------------------------------------------------
    # Road styling
    # --------------------------------------------------------

    def style_function(feature):

        p = feature["properties"]

        if p["blocked"]:
            return {
                "color": "#444444",
                "weight": 2,
                "opacity": 0.8,
                "dashArray": "8, 8"
            }

        density = p["density"]

        if density < 0.25:
            color = "green"

        elif density < 0.50:
            color = "yellow"

        elif density < 0.75:
            color = "orange"

        else:
            color = "red"

        return {
            "color": color,
            "weight": 2,
            "opacity": 0.7
        }

    # --------------------------------------------------------
    # GeoJSON
    # --------------------------------------------------------

    folium.GeoJson(
        geojson_data,
        name="Traffic network",
        style_function=style_function,
        show=True,
        overlay=True,
        control=True
    ).add_to(m)

    # --------------------------------------------------------
    # Layer control
    # --------------------------------------------------------

    folium.LayerControl(
        collapsed=False
    ).add_to(m)

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    m.save(output_file)

    print(
        f"\nGeoJSON map saved to: {output_file}"
    )

def density_color(density):
    if density < 0.25:
        return "green"
    elif density < 0.50:
        return "yellow"
    elif density < 0.75:
        return "orange"
    else:
        return "red"

def test1():
    # ---------- 1. Test the parser ----------
    assert parse_maxspeed(None) == 40
    assert parse_maxspeed("50") == 50
    assert parse_maxspeed("50;60") == 50
    assert parse_maxspeed(["50", "40"]) == 40
    assert abs(parse_maxspeed("40 mph") - 64.36) < 0.1
    assert parse_maxspeed("walk") == 40
    print("parse_maxspeed: all tests passed\n")

    # ---------- 2. Load a small region (~0.9 km x 0.9 km) ----------
    small_bbox = (106.694, 10.769, 106.702, 10.777)   # left, bottom, right, top
    g = load_graph(small_bbox)

    n_edges = sum(len(t) for t in g.adj.values())
    n_oneway = sum(e.oneway for t in g.adj.values() for e in t.values())
    print(f"{len(g.vertices)} vertices, {n_edges} directed edges "
        f"({n_oneway} one-way), v_max = {g.v_max:.0f} km/h\n")

    # ---------- 3. Print everything ----------
    print("VERTICES")
    for v in g.vertices.values():
        print(f"  {v.id}: lat={v.y:.6f}, lon={v.x:.6f}")

    print("\nEDGES")
    for u in g.adj:
        for e in g.adj[u].values():
            print(f"  {e.source} -> {e.target} | {e.length:7.1f} m | "
                f"{e.speed_limit:4.0f} km/h | oneway={e.oneway} | "
                f"pts={len(e.geometry)} | {e.name or '-'}")

    # ---------- 4. Try the computed properties ----------
    e = next(iter(next(iter(g.adj.values())).values()), None) or \
        next(x for t in g.adj.values() for x in t.values())
    print(f"\nSample edge {e.source}->{e.target}: "
        f"free-flow {e.travel_time:.1f} s", end="")
    e.density, e.weather_factor = 0.5, 0.8
    print(f", with density 0.5 and weather 0.8: {e.current_speed:.1f} km/h, "
        f"{e.travel_time:.1f} s")
    e.density, e.weather_factor = 0.0, 1.0   # reset

    # ---------- 5. Generate the scenario ----------
    cfg = ScenarioConfig(
        base_density=0.1,
        max_density=0.9,
        density_deviation=0.05,
        block_probability=0.05,

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

    n_blocked = sum(
        e.blocked
        for t in g.adj.values()
        for e in t.values()
    )

    print(f"Blocked edges: {n_blocked} of {n_edges}")

    for u in g.adj:
        for v, e in g.adj[u].items():
            # A two-way road is blocked in both directions or neither
            if not e.oneway:
                assert g.adj[v][u].blocked == e.blocked

    densities = [
        e.density
        for t in g.adj.values()
        for e in t.values()
    ]

    print(
        f"Density: min={min(densities):.2f}, "
        f"max={max(densities):.2f}, "
        f"avg={sum(densities) / len(densities):.2f}"
    )

    edges = [
        e
        for t in g.adj.values()
        for e in t.values()
    ]

    edges.sort(key=lambda e: e.density, reverse=True)

    print("\nMost congested roads:")
    for e in edges[:10]:
        lat, lon = road_midpoint(e.geometry)

        print(
            f"  {e.source}->{e.target} | "
            f"density={e.density:.2f} | "
            f"midpoint=({lat:.6f}, {lon:.6f}) | "
            f"{e.name or '-'}"
        )

    # ---------- 6. Map built only from Graph data ----------
    cx = sum(v.x for v in g.vertices.values()) / len(g.vertices)
    cy = sum(v.y for v in g.vertices.values()) / len(g.vertices)

    m = folium.Map(location=[cy, cx], zoom_start=17, tiles=None)

    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles © Esri",
        name="Esri Streets",
        max_zoom=19,
    ).add_to(m)

    # green (free) -> yellow -> red (jam), with a legend on the map
    cmap = LinearColormap(
        ["#1a9850", "#fee08b", "#d73027"],
        vmin=cfg.base_density, vmax=cfg.max_density,
        caption="Traffic density (green = free flow, red = jam)",
    )
    cmap.add_to(m)

    def density_color(d: float) -> str:
        return cmap(d)[:7]                     # strip the alpha channel

    # red circle with a white bar = "no entry"
    NO_ENTRY_HTML = (
        '<div style="box-sizing:border-box;width:18px;height:18px;border-radius:50%;'
        'background:#d00;border:2px solid white;box-shadow:0 0 3px rgba(0,0,0,.7);'
        'display:flex;align-items:center;justify-content:center;">'
        '<div style="width:9px;height:3px;background:white;"></div></div>'
    )

    closed_layer = folium.FeatureGroup(name="Closed roads").add_to(m)   # can be toggled on/off

    for u in g.adj:
        for e in g.adj[u].values():
            if not e.oneway and e.source > e.target:
                continue                       # two-way roads share one state, draw once

            if e.blocked:
                label = (f"CLOSED | {e.name or '-'} | "
                        f"{'one-way' if e.oneway else 'two-way'} | {e.length:.0f} m")
                folium.PolyLine(e.geometry, color="#444444", weight=5, opacity=0.9,
                                dash_array="4 8", tooltip=label).add_to(closed_layer)
                folium.Marker(road_midpoint(e.geometry),
                            icon=folium.DivIcon(html=NO_ENTRY_HTML,
                                                icon_size=(18, 18), icon_anchor=(9, 9)),
                            tooltip=label).add_to(closed_layer)
                continue

            color = density_color(e.density)
            line = folium.PolyLine(
                e.geometry,
                color=color,
                weight=4,
                opacity=0.9,
                tooltip=(
                    f"{e.name or '-'} | "
                    f"{'one-way' if e.oneway else 'two-way'} | "
                    f"density {e.density:.2f} | "
                    f"weather {e.weather_factor:.2f} | "
                    f"{e.current_speed:.0f} km/h | "
                    f"{e.length:.0f} m, {e.travel_time:.0f} s"
                )
            ).add_to(m)
            if e.oneway:                       # arrows show the allowed direction
                PolyLineTextPath(line, "   ►   ", repeat=True, offset=6,
                                attributes={"fill": color, "font-size": "14"}).add_to(m)

    for v in g.vertices.values():
        folium.CircleMarker([v.y, v.x], radius=2, color="black",
                            tooltip=str(v.id)).add_to(m)

    rain_pane = folium.map.CustomPane(
        "rainPane",
        z_index=350
    ).add_to(m)

    m.get_root().header.add_child(folium.Element("""
    <style>
    .leaflet-rainPane-pane {
        pointer-events: none !important;
    }
    </style>
    """))

    rain_layer = folium.FeatureGroup(
        name="Rain zones",
        show=True
    ).add_to(m)

    for cell in cfg.rain_cells:
        bounds = [
            [cell.min_lat, cell.min_lon],
            [cell.max_lat, cell.max_lon]
        ]

        folium.Rectangle(
            bounds=bounds,
            color="#6baed6",
            weight=2,
            fill=True,
            fill_color="#9ecae1",
            fill_opacity=0.25,
            interactive=False,
            pane="rainPane"
        ).add_to(rain_layer)

    legend_html = """
    <div style="position: fixed; bottom: 24px; left: 12px; z-index: 9999;
                background: white; padding: 8px 10px;
                border: 1px solid #999; border-radius: 4px;
                font-size: 13px;">

    <div>
        <span style="display:inline-block;width:30px;border-top:4px dashed #444;
            vertical-align:middle;"></span>
        &nbsp;Closed road (incident)
    </div>

    <div>
        <span style="display:inline-block;width:24px;height:12px;
            background:#9ecae1;border:2px solid #6baed6;
            vertical-align:middle;"></span>
        &nbsp;Rain zone
    </div>

    <div>
        &#9658; arrows = direction of a one-way road
    </div>

    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    folium.LayerControl().add_to(m)
    m.save("graph_test_map.html")
    print("\nSaved graph_test_map.html")

def test2():
    """
    Test 2:
    Load and stress-test a larger (~5 km x 5 km)
    road network without rendering every road.
    """

    print("=" * 60)
    print("TEST 2 — LARGE GRAPH LOAD TEST")
    print("=" * 60)

    # ========================================================
    # 1. Load larger graph
    # ========================================================

    bbox = (
        106.690,
        10.750,
        106.740,
        10.795
    )

    print("\nLoading OSM graph...")

    start = time.perf_counter()

    g = load_graph(bbox)

    load_time = time.perf_counter() - start

    # ========================================================
    # 2. Graph statistics
    # ========================================================

    n_vertices = len(g.vertices)

    n_edges = sum(
        len(targets)
        for targets in g.adj.values()
    )

    n_oneway = sum(
        edge.oneway
        for targets in g.adj.values()
        for edge in targets.values()
    )

    print(
        f"\nGraph loaded in {load_time:.2f} s"
    )

    print("\nGraph statistics:")
    print(f"  Vertices:       {n_vertices:,}")
    print(f"  Directed edges: {n_edges:,}")
    print(f"  One-way edges:  {n_oneway:,}")
    print(f"  v_max:          {g.v_max:.0f} km/h")


    # ========================================================
    # 3. Generate scenario
    # ========================================================

    cfg = ScenarioConfig(
        base_density=0.1,
        max_density=0.9,
        density_deviation=0.05,
        block_probability=0.05,

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
                min_lat=10.7715,
                max_lat=10.7745,
                min_lon=106.6965,
                max_lon=106.6995,
                weather_factor=0.6
            )
        ]
    )

    print("\nGenerating scenario...")

    start = time.perf_counter()

    ScenarioGenerator(
        seed=424545,
        config=cfg
    ).apply(g)

    scenario_time = time.perf_counter() - start

    print(
        f"Scenario generated in "
        f"{scenario_time:.2f} s"
    )


    # ========================================================
    # 4. Validate two-way roads
    # ========================================================

    print("\nValidating two-way roads...")

    two_way_count = 0

    for u in g.adj:

        for v, edge in g.adj[u].items():

            if not edge.oneway:

                two_way_count += 1

                reverse = g.adj[v][u]

                assert reverse.blocked == edge.blocked

                assert abs(
                    reverse.density - edge.density
                ) < 1e-12

    print(
        f"  Checked {two_way_count:,} "
        f"two-way directions"
    )

    print(
        "  Validation: PASSED"
    )


    # ========================================================
    # 5. Scenario statistics
    # ========================================================

    edges = [
        edge
        for targets in g.adj.values()
        for edge in targets.values()
    ]

    densities = [
        edge.density
        for edge in edges
    ]

    blocked = sum(
        edge.blocked
        for edge in edges
    )

    print("\nScenario statistics:")

    print(
        f"  Density min:     "
        f"{min(densities):.3f}"
    )

    print(
        f"  Density max:     "
        f"{max(densities):.3f}"
    )

    print(
        f"  Density average: "
        f"{sum(densities) / len(densities):.3f}"
    )

    print(
        f"  Blocked edges:    "
        f"{blocked:,} / {len(edges):,}"
    )

    print(
        f"  Blocked ratio:    "
        f"{100 * blocked / len(edges):.2f}%"
    )


    # ========================================================
    # 6. Congestion statistics
    # ========================================================

    edges.sort(
        key=lambda edge: edge.density,
        reverse=True
    )

    print("\nTop 10 congested roads:")

    for edge in edges[:10]:

        lat, lon = road_midpoint(
            edge.geometry
        )

        print(
            f"  density={edge.density:.2f} | "
            f"{edge.name or '-'} | "
            f"({lat:.5f}, {lon:.5f})"
        )

    # ========================================================
    # 7. GeoJSON visualization
    # ========================================================

    print("\nCreating GeoJSON visualization...")

    start = time.perf_counter()

    create_geojson_map(
        g,
        "test2_large_geojson.html"
    )

    map_time = time.perf_counter() - start

    print(
        f"GeoJSON map generated in "
        f"{map_time:.2f} s"
    )

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
        local_vmax=False
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

if __name__ == "__main__":
    test3()
