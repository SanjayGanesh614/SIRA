# import folium

# # Example buoy data: lon, lat, and some info
# buoys = [
#     {"lon": 72.5, "lat": 5.0, "data": "Buoy 1: Temp=28°C, Salinity=35 PSU"},
#     {"lon": 75.0, "lat": -2.0, "data": "Buoy 2: Temp=26°C, Salinity=34.8 PSU"},
#     {"lon": 80.0, "lat": 10.0, "data": "Buoy 3: Temp=29°C, Salinity=35.1 PSU"},
# ]

# # Create map centered in Indian Ocean
# m = folium.Map(location=[0, 75], zoom_start=4)

# # Add buoy markers with popups
# for buoy in buoys:
#     folium.Marker(
#         location=[buoy["lat"], buoy["lon"]],
#         popup=folium.Popup(buoy["data"], max_width=200)
#     ).add_to(m)

# # Save to HTML
# m.save("buoy_map.html")

import pandas as pd
import folium

# Load CSV
df = pd.read_csv("argo_semantic_summary_1.csv")

# Keep only unique lat-lon combinations with region
unique_locations = df[['latitude', 'longitude', 'region']].drop_duplicates()

# Create Folium map centered roughly in the Indian Ocean
m = folium.Map(location=[0, 80], zoom_start=3)

# Add markers
for _, row in unique_locations.iterrows():
    folium.Marker(
        location=[row['latitude'], row['longitude']],
        popup=f"{row['region']} ({row['latitude']:.2f}, {row['longitude']:.2f})"
    ).add_to(m)

# Save map as HTML
m.save("buoy_map.html")
print("Map saved as buoy_map.html. Open in browser to view.")
