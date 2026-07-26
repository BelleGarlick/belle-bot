from dataclasses import dataclass

from pyproj import Transformer

# Initialize transformer: EPSG:4326 (WGS84 Lat/Lon) to EPSG:32630 (UTM Zone 30N - covers London)
# Choose the correct UTM zone EPSG code based on your geographic location!
transformer = Transformer.from_crs("EPSG:4326", "EPSG:32630", always_xy=True)


@dataclass
class GpsPoint:
    timestamp: float
    x: float
    y: float
    altitude: float

    @staticmethod
    def from_data(timestamp, data: dict):
        x, y = transformer.transform(data['longitude'], data['latitude'])
        return GpsPoint(
            timestamp=timestamp,
            x=x,
            y=y,
            altitude=float(data["altitude"])
        )
