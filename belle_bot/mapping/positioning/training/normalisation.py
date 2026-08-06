import json

import numpy as np

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
            self.normalisation_bounds[key] = float(value)

        self.normalisation_bounds[key] = max(self.normalisation_bounds[key], float(value))

    def fit(self, replays):
        for x in replays:
            first_gps_pos: GpsPoint = [item for item in x.events if isinstance(item, GpsPoint)][0]
            x0, y0, alt0 = first_gps_pos.x, first_gps_pos.y, first_gps_pos.altitude

            for item in x.events:
                if isinstance(item, ImuData):
                    # Update normalisation if possible
                    self.update("imu.acc", np.max(np.abs(item.acc)))
                    self.update("imu.gyro", np.max(np.abs(item.gyro)))
                    self.update("imu.angle", np.max(np.abs(item.angle)))

                elif isinstance(item, GpsPoint):
                    delta_x = item.x - x0
                    delta_y = item.y - y0
                    delta_alt = item.altitude - alt0
                    self.update("gps.x", abs(delta_x))
                    self.update("gps.y", abs(delta_y))
                    self.update("gps.alt", abs(delta_alt))

                else:
                    # split up camera into multiple tokens
                    raise NotImplementedError()

        return self

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.normalisation_bounds, f)

        return self

    def load(self, path):
        with open(path, "r") as f:
            self.normalisation_bounds = json.load(f)

        return self
