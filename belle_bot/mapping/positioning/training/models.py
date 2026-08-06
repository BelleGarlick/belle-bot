import base64
from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np
from pyproj import Transformer

# Initialize transformer: EPSG:4326 (WGS84 Lat/Lon) to EPSG:32630 (UTM Zone 30N - covers London)
# Choose the correct UTM zone EPSG code based on your geographic location!
transformer = Transformer.from_crs("EPSG:4326", "EPSG:32630", always_xy=True)


class ModalityEnum(int, Enum):
    PAD = 0
    IMU = 1
    GPS = 2


@dataclass
class GpsPoint:
    timestamp: float
    x: float
    y: float
    altitude: float
    name: str | None = None

    @staticmethod
    def from_data(timestamp, data: dict):
        x, y = transformer.transform(data['longitude'], data['latitude'])
        return GpsPoint(
            timestamp=timestamp,
            x=x,
            y=y,
            altitude=float(data["altitude"])
        )

    def numpy(self) -> np.ndarray:
        return np.array([self.x, self.y, self.altitude])

    def copy(self):
        return GpsPoint(**self.__dict__)



@dataclass
class ImuData:
    timestamp: int
    gyro: np.ndarray
    acc: np.ndarray
    angle: np.ndarray
    name: str | None = None

    @staticmethod
    def from_data(timestamp, data):
        parse_datum = lambda key: np.frombuffer(base64.b64decode(data[key]), dtype=np.float32)

        return ImuData(
            timestamp=timestamp,
            acc=parse_datum("acc"),
            gyro=parse_datum("gyro"),
            angle=parse_datum("angle"),
        )

    def copy(self):
        return ImuData(**self.__dict__)



@dataclass
class CameraData:
    timestamp: int
    frame: np.ndarray

    @staticmethod
    def from_data(timestamp, data):
        frame = cv2.imdecode(np.frombuffer(base64.b64decode(data['rgb']), np.uint8), cv2.IMREAD_COLOR)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        return CameraData(
            timestamp=timestamp,
            frame=frame,
        )

    def copy(self):
        return CameraData(**self.__dict__)
