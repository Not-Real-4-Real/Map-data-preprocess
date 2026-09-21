import math
import re
import random
from dataclasses import dataclass, field

import osmnx as ox

DEFAULT_SPEED = 40          # km/h, used when maxspeed is missing
MIN_SPEED_FACTOR = 0.05     # keeps current_speed > 0 so travel_time never divides by zero

@dataclass
class Vertex:
    id: int
    x: float   # longitude
    y: float   # latitude

@dataclass
class Edge:
    source: int
    target: int
    length: float                  # meters
    speed_limit: float             # km/h
    oneway: bool = False
    name: str = ""
    geometry: list = field(default_factory=list)   # [(lat, lon), ...] for drawing
    density: float = 0.0           # 0 (empty) .. 1 (jammed)
    weather_factor: float = 1.0    # 0..1
    incident_factor: float = 1.0   # 0..1
    blocked: bool = False

    @property
    def current_speed(self) -> float:
        """km/h. Placeholder model: speed drops linearly with density
        (Greenshields), then weather/incident multipliers apply."""
        speed = self.speed_limit * (1 - self.density)
        speed *= self.weather_factor * self.incident_factor
        return max(speed, self.speed_limit * MIN_SPEED_FACTOR)

    @property
    def travel_time(self) -> float:
        """Seconds; infinite if the edge is blocked."""
        if self.blocked:
            return math.inf
        return self.length / (self.current_speed / 3.6)

class Graph:
    def __init__(self):
        self.vertices = {}   # id -> Vertex
        self.adj = {}        # u -> {v: Edge}   (forward search)
        self.radj = {}       # v -> {u: Edge}   (backward search)
        self.v_max = 0.0     # max speed limit in the graph, km/h

    def add_vertex(self, v: Vertex):
        self.vertices[v.id] = v
        self.adj.setdefault(v.id, {})
        self.radj.setdefault(v.id, {})

    def add_edge(self, e: Edge):
        self.adj[e.source][e.target] = e
        self.radj[e.target][e.source] = e
        self.v_max = max(self.v_max, e.speed_limit)

    def neighbors(self, u):          # outgoing edges of u
        return self.adj[u].values()

    def predecessors(self, v):       # incoming edges of v (for the backward search)
        return self.radj[v].values()

    def get_edge(self, u, v):
        return self.adj[u].get(v)

def parse_maxspeed(raw, default = DEFAULT_SPEED):
    if raw is None:
        return default
    items = raw if isinstance(raw, list) else [raw]
    speeds = []
    for item in items:
         for part in str(item).split(";"):
            m = re.match(r"\s*(\d+(?:\.\d+)?)\s*(mph)?", part)
            if m:
                v = float(m.group(1))
                speeds.append(v * 1.609 if m.group(2) else v)
    return min(speeds) if speeds else default

def parse_name(raw) -> str:
    if raw is None:
        return ""
    return " / ".join(raw) if isinstance(raw, list) else str(raw)

def load_graph(bbox) -> Graph:
    G = ox.graph_from_bbox(bbox, network_type="drive")
    g = Graph()

    for nid, d in G.nodes(data=True):
        g.add_vertex(Vertex(nid, d["x"], d["y"]))

    for u, v, _key, d in G.edges(keys=True, data=True):
        if u == v:                       # skip self-loops
            continue
        geom = d.get("geometry")
        if geom is not None:
            coords = [(lat, lon) for lon, lat in geom.coords]
        else:
            coords = [(g.vertices[u].y, g.vertices[u].x),
                      (g.vertices[v].y, g.vertices[v].x)]
        edge = Edge(
            source=u,
            target=v,
            length=float(d["length"]),
            speed_limit=parse_maxspeed(d.get("maxspeed")),
            name=parse_name(d.get("name")),
            geometry=coords,
        )
        old = g.get_edge(u, v)           # parallel edges: keep the shortest
        if old is None or edge.length < old.length:
            g.add_edge(edge)

    for u, targets in g.adj.items():     # one-way = no edge going back
        for v, e in targets.items():
            e.oneway = u not in g.adj[v]
    return g

@dataclass
class ScenarioConfig:
    min_density: float = 0.0
    max_density: float = 0.9   # keep below 1 so the speed floor is rarely hit
    block_probability: float = 0.05

def _roads(graph):
    """Yield (edge, twin) once per physical road.
    twin is the opposite direction of a two-way road, or None for a one-way road."""
    for u in sorted(graph.adj):
        for v in sorted(graph.adj[u]):
            edge = graph.adj[u][v]
            if not edge.oneway and v < u:
                continue                  # already handled from the other direction
            yield edge, (None if edge.oneway else graph.adj[v][u])


class ScenarioGenerator: #randomly assign density to edge
    def __init__(self, seed: int, config: ScenarioConfig | None = None):
        self.seed = seed
        self.config = config or ScenarioConfig()

    def apply(self, graph: Graph) -> None:
        """One call = one snapshot. Overwrites any earlier scenario on the graph."""
        self._assign_density(graph)
        self._assign_incidents(graph)

    def _assign_density(self, graph: Graph) -> None:
        rng = random.Random(self.seed)
        cfg = self.config
        for edge, twin in _roads(graph):
            edge.density = rng.uniform(cfg.min_density, cfg.max_density)
            if twin:
                twin.density = edge.density

    def _assign_incidents(self, graph: Graph) -> None:
        rng = random.Random(f"incidents-{self.seed}")   # separate random stream
        for edge, twin in _roads(graph):
            edge.blocked = rng.random() < self.config.block_probability
            if twin:
                twin.blocked = edge.blocked

def haversine_m(lat1, lon1, lat2, lon2) -> float:
    """Great-circle distance in meters."""
    R = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))

import folium
from folium.plugins import PolyLineTextPath
from branca.colormap import LinearColormap  

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
cfg = ScenarioConfig(min_density=0.0, max_density=0.9, block_probability=0.05)
ScenarioGenerator(seed=424545, config=cfg).apply(g)

n_blocked = sum(e.blocked for t in g.adj.values() for e in t.values())
print(f"Blocked edges: {n_blocked} of {n_edges}")

for u in g.adj:                            # a two-way road is blocked in both directions or neither
    for v, e in g.adj[u].items():
        if not e.oneway:
            assert g.adj[v][u].blocked == e.blocked

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
    vmin=cfg.min_density, vmax=cfg.max_density,
    caption="Traffic density (green = free flow, red = jam)",
)
cmap.add_to(m)

def density_color(d: float) -> str:
    return cmap(d)[:7]                     # strip the alpha channel

def road_midpoint(geometry):
    """A point in the middle of the drawn road, used to place the closure sign."""
    n = len(geometry)
    if n % 2 == 1:
        return geometry[n // 2]
    (lat1, lon1), (lat2, lon2) = geometry[n // 2 - 1], geometry[n // 2]
    return ((lat1 + lat2) / 2, (lon1 + lon2) / 2)

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
            tooltip=(f"{e.name or '-'} | {'one-way' if e.oneway else 'two-way'} | "
                     f"density {e.density:.2f} | {e.current_speed:.0f} km/h | "
                     f"{e.length:.0f} m, {e.travel_time:.0f} s"),
        ).add_to(m)
        if e.oneway:                       # arrows show the allowed direction
            PolyLineTextPath(line, "   ►   ", repeat=True, offset=6,
                             attributes={"fill": color, "font-size": "14"}).add_to(m)

for v in g.vertices.values():
    folium.CircleMarker([v.y, v.x], radius=2, color="black",
                        tooltip=str(v.id)).add_to(m)

legend_html = """
<div style="position: fixed; bottom: 24px; left: 12px; z-index: 9999; background: white;
            padding: 8px 10px; border: 1px solid #999; border-radius: 4px; font-size: 13px;">
  <div><span style="display:inline-block;width:30px;border-top:4px dashed #444;
        vertical-align:middle;"></span>&nbsp;Closed road (incident)</div>
  <div>&#9658; arrows = direction of a one-way road</div>
</div>
"""
m.get_root().html.add_child(folium.Element(legend_html))

folium.LayerControl().add_to(m)
m.save("graph_test_map.html")
print("\nSaved graph_test_map.html")