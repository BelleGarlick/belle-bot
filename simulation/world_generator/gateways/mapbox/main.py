import math
import requests
import mapbox_vector_tile
from matplotlib import pyplot as plt


# todo save the api requests this map so i can pass it when offline


def lat_lng_to_tile_xy(lat, lng, zoom):
  n = math.pow(2, zoom)
  x = math.floor(n * ((lng + 180) / 360))
  lat_rad = (lat * math.pi) / 180
  y = math.floor(
    n * (1 - math.log(math.tan(lat_rad) + 1 / math.cos(lat_rad)) / math.pi) / 2
  )
  
  return x, y, zoom


def get_vector_tile(x, y, zoom, tileset_id="mapbox.mapbox-streets-v8", format="vector.pbf"):
    endpoint = f"https://api.mapbox.com/v4/{tileset_id}/{zoom}/{x}/{y}.{format}?access_token={access_token}"
    response = requests.get(endpoint)
    return response


def parse_vector_tile(content):
    """
    Parses the binary vector tile content into a dictionary.
    """
    return mapbox_vector_tile.decode(content)


if __name__ == "__main__":
    # Example usage for London at zoom level 12
    x, y, zoom = lat_lng_to_tile_xy(51.535385, -0.099991, 15)
    response = get_vector_tile(x, y, zoom)

    if response.status_code == 200:
        decoded_data = parse_vector_tile(response.content)

        for layer, data in decoded_data.items():
            if layer == 'road':
                for feature in data['features']:
                    geometry = feature['geometry']
                    properties = feature['properties']

                    coordinates = geometry['coordinates']

                    print(properties)

                    if isinstance(coordinates[0], int):
                        plt.scatter(coordinates[0], coordinates[1])

                    elif isinstance(coordinates[0][0], int):
                        plt.plot([x[0] for x in coordinates], [x[1] for x in coordinates])

                    else:
                        for coords in coordinates:
                            plt.plot([x[0] for x in coords], [x[1] for x in coords])


            elif layer == 'building':
                pass
                # breakpoint()

            elif layer == 'landuse':
                pass
                # breakpoint()

            elif layer == 'natural_label':
                pass
                # breakpoint()

            elif layer == 'structure':
                pass
                # breakpoint()

            elif layer == 'water':
                pass
                # breakpoint()

            elif layer == 'waterway':
                pass
                # breakpoint()

            else:
                print(f"Not handling layer {layer}")

        plt.axis("equal")
        plt.show()

    else:
        print(f"Failed to fetch tile: {response.status_code}")
