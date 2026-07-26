import numpy as np

from belle_bot.mapping.positioning.training.create_dataset import ImuData
from belle_bot.mapping.positioning.training.models import GpsPoint


class NormalisationBounds:
    def __init__(self):
        self.trainable = True
        self.normalisation_bounds = {}

    def __getitem__(self, item):
        return self.normalisation_bounds[item]

    def __setitem__(self, key, value):
        self.normalisation_bounds[key] = value

    def update(self, key, value):
        if key not in self.normalisation_bounds:
            self.normalisation_bounds[key] = value

        self.normalisation_bounds[key] = max(self.normalisation_bounds[key], value)

    def fit(self, replays):
        for x in replays:
            first_gps_pos: GpsPoint = x.gps[0]
            x0, y0, alt0 = first_gps_pos.x, first_gps_pos.y, first_gps_pos.altitude

            for item in x.gps + x.imu:
                if isinstance(item, ImuData):
                    # Update normalisation if possible
                    self.update("imu.acc", np.max(np.abs(item.acc)))
                    self.update("imu.gyro", np.max(np.abs(item.gyro)))
                    self.update("imu.angle", np.max(np.abs(item.angle)))

                elif isinstance(item, GpsPoint):
                    delta_x = item.x - x0
                    delta_y = item.y - y0
                    self.update("gps.x", abs(delta_x))
                    self.update("gps.y", abs(delta_y))

                else:
                    # split up camera into multiple tokens
                    raise NotImplementedError()

            return self

