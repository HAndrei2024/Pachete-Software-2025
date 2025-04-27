import geopandas as gpd
import numpy as np
import osmnx as ox
import requests
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
from shapely.geometry import Point
import contextily

url = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
countries = gpd.read_file(url)

'''
usa = usa[usa["ADMIN"] == "United States of America"]

usa.to_file("usa_boundary.shp", driver="ESRI Shapefile")

cities_url = "https://naciscdn.org/naturalearth/10m/cultural/ne_10m_populated_places.zip"
cities = gpd.read_file(cities_url)
cities = cities[cities["SOV0NAME"] == "United States"]
cities.to_file("usa_cities_detailed.shp", driver="ESRI Shapefile")

rivers = ox.features_from_place("New York, United States of America", {"waterway": "river"})
rivers.to_file("new_york_rivers.shp", driver="ESRI Shapefile")

print("Shapefiles saved: usa_boundary.shp, usa_cities_detailed.shp, new_york_rivers.shp")
'''

usa = gpd.read_file("geopandas_files/usa_boundary.shp")
cities = gpd.read_file("geopandas_files/usa_cities_detailed.shp")
rivers = gpd.read_file("geopandas_files/new_york_rivers.shp")

# Afisare harta SUA cu orase, raurile din New York

fig, ax = plt.subplots(figsize=(12, 10))
usa.plot(ax=ax, color="lightgray", edgecolor="black")
rivers.plot(ax=ax, color="blue", linewidth=0.7, label="Rivers")
cities.plot(ax=ax, color="red", markersize=10, label="Cities")

plt.title("Map of USA - Boundaries, Rivers, and Cities", fontsize=14)
plt.legend()
plt.show()

usa_1 = countries[countries["ADMIN"] == "United States of America"]

# Selectare vecini (tarile care ating) SUA

neighbors = countries[countries.touches(usa_1.geometry.iloc[0])]

usa_neighbors = pd.concat([usa, neighbors])

# Afisare harta vecini SUA

fig, ax = plt.subplots(figsize=(10, 8))
neighbors.plot(ax=ax, color="lightblue", edgecolor="black", label="Neighbors")
usa.plot(ax=ax, color="lightgreen", edgecolor="black", label="USA")

plt.title("USA and Neighbors")
plt.legend()
plt.show()

neighbors_utm = neighbors.to_crs(epsg=3857)

# Coordonatele Washington
washington_wgs = Point(-77.0369, 38.9072)
# Transformare in coordonate metrice
washington_utm = gpd.GeoSeries([washington_wgs], crs="EPSG:4326").to_crs(epsg=3857).iloc[0]

# Calcul centroizi
neighbors_utm["centroid"] = neighbors_utm.geometry.centroid

# Distanta de la Washington la centroizi
neighbors_utm["distance_m"] = neighbors_utm["centroid"].distance(washington_utm)
neighbors_utm["distance_km"] = neighbors_utm["distance_m"] / 1000

print(neighbors_utm[["ADMIN", "distance_km"]].sort_values("distance_km"))


usa_neighbors["name"] = usa_neighbors["ADMIN"]

# Convertim în coordonate metrice metric EPSG:3857 (pentru contextily)
usa_neighbors = usa_neighbors.to_crs(epsg=3857)

# Coordonatele Washington
washington = gpd.GeoDataFrame(
    {'name': ['Washington D.C.']},
    geometry=[Point(-77.0369, 38.9072)],
    crs="EPSG:4326"
).to_crs(epsg=3857)

# Calcul centroizi
usa_neighbors["centroid"] = usa_neighbors.geometry.centroid

ax = usa_neighbors.plot(
    column='name',
    figsize=(12, 8),
    alpha=0.5,
    edgecolor='black',
    legend=True
)

usa_neighbors["centroid"].plot(ax=ax, color="yellow", markersize=30)

# Punct Washington
washington.plot(ax=ax, color='black', marker='*', markersize=100)

# Fundal geografic
contextily.add_basemap(ax, crs=usa_neighbors.crs.to_string())

plt.title("USA and its Neighbours", fontsize=16)
plt.axis("off")
plt.tight_layout()
plt.show()


# Afisare tari din America

world = gpd.read_file(url)

americas = world[world["CONTINENT"] == "North America"].copy()
south_america = world[world["CONTINENT"] == "South America"].copy()
americas = pd.concat([americas, south_america])

fig, ax = plt.subplots(figsize=(12, 8))
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)

americas.plot(
    cmap="viridis",
    linewidth=0.8,
    ax=ax,
    edgecolor="black",
    legend=True,
    cax=cax
)

for idx, row in americas.iterrows():
    if row["geometry"].geom_type in ["Polygon", "MultiPolygon"]:
        centroid = row["geometry"].centroid
        ax.text(centroid.x, centroid.y, row["ADMIN"], fontsize=8, ha='center')

plt.title("American Countries", fontsize=16)
plt.axis("off")
plt.tight_layout()
plt.show()