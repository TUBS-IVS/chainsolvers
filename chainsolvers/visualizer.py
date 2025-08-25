import uuid
import os
import folium
import networkx as nx
from pyproj import Transformer
import matplotlib.colors as mcolors
import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

class Visualizer:
    def __init__(self, savedir):
        self.savedir = savedir
        self.tree = nx.DiGraph()
        self.locations = {}  # node_id -> {coords, metadata, label, level}
        self.transformer = Transformer.from_crs(25832, 4326, always_xy=True)

    def add_node(self, parent_id, label, location=None, metadata=None):
        node_id = str(uuid.uuid4())
        self.tree.add_node(node_id, label=label)
        # Determine level
        if parent_id:
            self.tree.add_edge(parent_id, node_id)
            parent_level = self.locations[parent_id]["level"] if parent_id in self.locations else 0
            level = parent_level + 1
        else:
            level = 0
        # Always store node metadata, even if coords is None
        self.locations[node_id] = {
            "coords": location,
            "metadata": metadata or {},
            "label": label,
            "level": level
        }
        return node_id

    def visualize(self):
        if not self.locations:
            print("No locations to visualize.")
            return

        # Find first node with valid coordinates
        root_node = next((info for info in self.locations.values() if info["coords"] is not None), None)
        if not root_node:
            print("No valid coordinates found for visualization.")
            return

        lon, lat = self.transformer.transform(root_node["coords"][0], root_node["coords"][1])
        m = folium.Map(location=[lat, lon], zoom_start=13)

        # Set up colormap
        levels = [info["level"] for info in self.locations.values()]
        max_level = max(levels) if levels else 1
        from matplotlib import cm
        cmap = cm.get_cmap('viridis', max_level + 1)
        norm = mcolors.Normalize(vmin=0, vmax=max_level)

        # Add nodes (skip ones without coords)
        for node_id, info in self.locations.items():
            if info["coords"] is None:
                continue
            easting, northing = info["coords"]
            lon, lat = self.transformer.transform(easting, northing)
            color = mcolors.to_hex(cmap(norm(info["level"])))
            popup = (f"{info['label']}<br>"
                     f"Metadata: {info['metadata']}<br>"
                     f"Level: {info['level']}")
            folium.CircleMarker([lat, lon], radius=5, color=color, fill=True, popup=popup).add_to(m)

        # Add edges between parent and child nodes
        for parent_id, child_id in self.tree.edges():
            parent_info = self.locations.get(parent_id)
            child_info = self.locations.get(child_id)
            if parent_info and child_info:
                if parent_info["coords"] is None or child_info["coords"] is None:
                    continue
                if parent_info["level"] == 0:
                    continue  # Skip edge from root
                e1, n1 = parent_info["coords"]
                e2, n2 = child_info["coords"]
                lon1, lat1 = self.transformer.transform(e1, n1)
                lon2, lat2 = self.transformer.transform(e2, n2)
                folium.PolyLine([(lat1, lon1), (lat2, lon2)], color='black', weight=3).add_to(m)

        m.save("anchorchain_branching_map.html")
        print("Map saved as anchorchain_branching_map.html")

    def visualize_levels(self):
        if not self.locations:
            print("No locations to visualize.")
            return

        # Find root location for centering
        root_node = next((info for info in self.locations.values() if info["coords"] is not None), None)
        if not root_node:
            print("No valid coordinates found for visualization.")
            return

        lon, lat = self.transformer.transform(root_node["coords"][0], root_node["coords"][1] - 1000) # centre more south

        # Determine max level
        levels = [info["level"] for info in self.locations.values() if info["coords"] is not None]
        max_level = max(levels)

        from matplotlib import cm
        cmap = cm.get_cmap('viridis', max_level + 1)
        norm = mcolors.Normalize(vmin=0, vmax=max_level)

        # Group nodes by level
        level_groups = {lvl: [] for lvl in range(max_level + 1)}
        for node_id, info in self.locations.items():
            if info["coords"] is not None:
                level_groups[info["level"]].append((node_id, info))

        # Step 1: Individual level maps
        for lvl in range(max_level + 1):
            m = folium.Map(location=[lat, lon], zoom_start=13)
            for node_id, info in level_groups[lvl]:
                easting, northing = info["coords"]
                lon_, lat_ = self.transformer.transform(easting, northing)
                color = mcolors.to_hex(cmap(norm(info["level"])))
                popup = f"{info['label']}<br>Metadata: {info['metadata']}<br>Level: {info['level']}"
                folium.CircleMarker([lat_, lon_], radius=5, color=color, fill=True, popup=popup).add_to(m)

            for parent_id, child_id in self.tree.edges():
                p_info = self.locations.get(parent_id)
                c_info = self.locations.get(child_id)
                if p_info and c_info and p_info["coords"] is not None and c_info["coords"] is not None:
                    if p_info["level"] == lvl or c_info["level"] == lvl:
                        e1, n1 = p_info["coords"]
                        e2, n2 = c_info["coords"]
                        lon1, lat1 = self.transformer.transform(e1, n1)
                        lon2, lat2 = self.transformer.transform(e2, n2)
                        folium.PolyLine([(lat1, lon1), (lat2, lon2)], color='gray', weight=1).add_to(m)

            m.save(os.path.join(self.savedir, f"anchorchain_map_level_{lvl}.html"))
            logger.info(f"Map for level {lvl} saved in {self.savedir} as anchorchain_map_level_{lvl}.html")

        # Step 2: Cumulative level maps
        for lvl in range(max_level + 1):
            m = folium.Map(location=[lat, lon], zoom_start=13)


            for parent_id, child_id in self.tree.edges():
                p_info = self.locations.get(parent_id)
                c_info = self.locations.get(child_id)
                if p_info and c_info and p_info["coords"] is not None and c_info["coords"] is not None:
                    if p_info["level"] == 0:
                        continue  # Skip edge from root
                    if p_info["level"] <= lvl and c_info["level"] <= lvl:
                        e1, n1 = p_info["coords"]
                        e2, n2 = c_info["coords"]
                        lon1, lat1 = self.transformer.transform(e1, n1)
                        lon2, lat2 = self.transformer.transform(e2, n2)
                        folium.PolyLine([(lat1, lon1), (lat2, lon2)], color='black', weight=2).add_to(m)
            for l in range(lvl + 1):
                for node_id, info in level_groups[l]:
                    easting, northing = info["coords"]
                    lon_, lat_ = self.transformer.transform(easting, northing)
                    color = mcolors.to_hex(cmap(norm(info["level"])))
                    popup = f"{info['label']}<br>Metadata: {info['metadata']}<br>Level: {info['level']}"
                    folium.CircleMarker(
                        [lat_, lon_],
                        radius=8,  # larger marker
                        color='black',  # border color
                        fill=True,
                        fill_color=color,  # internal color from colormap
                        fill_opacity=1.0,  # fully opaque
                        weight=1,  # border thickness
                        popup=popup
                    ).add_to(m)
            m.save(os.path.join(self.savedir, f"anchorchain_map_levels_0_to_{lvl}.html"))
            logger.info(f"Map for levels 0 to {lvl} saved in {self.savedir} as anchorchain_map_levels_0_to_{lvl}.html")
