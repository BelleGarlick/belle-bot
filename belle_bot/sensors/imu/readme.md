# IMU Sensor Service

This service provides an interface for reading and processing data from an Inertial Measurement Unit (IMU) over a serial connection. It decodes acceleration, angular velocity, and Euler angles, and publishes the processed data to the Belle Bot fabric.

## Overview

The IMU service connects to a serial port, parses incoming data packets, and aggregates readings at a 10Hz frequency. The aggregated data is then published as odometry/sensor information.

## Key Features

- **Serial Communication**: Asynchronous reading from the serial port using `pyserial-asyncio`.
- **Data Decoding**: Supports decoding of:
  - Acceleration (Ax, Ay, Az)
  - Angular Velocity (Wx, Wy, Wz)
  - Euler Angles (Roll, Pitch, Yaw)
- **Data Aggregation**: Collects high-frequency raw data and publishes 10Hz averages to reduce noise and network load.
- **Fabric Integration**: Publishes data to the `sensors/imu` topic on the Belle Bot fabric.

## Requirements

- Python 3.x
- `numpy`
- `pyserial-asyncio`
- `belle_bot.fabric` (Internal dependency)

## Configuration

The following constants in `service.py` can be adjusted:

- `PORT`: The serial port address (default: `/dev/cu.usbserial-130`).
- `BAUDRATE`: The baud rate for serial communication (default: `9600`).
- `HZ_10_INTERVAL`: The publishing interval (default: `0.1` seconds for 10Hz).
- `FABRIC_SERVICE_ID`: The topic ID for publishing (default: `sensors/odometry`).

## Usage

To run the IMU service:

```bash
python -m belle_bot.sensors.imu.service
```

## Implementation Details

### `IMUProcessor` Class

The core logic is contained within the `IMUProcessor` class:
- `read_loop`: Continuously reads from the serial stream, validates checksums, and decodes packets.
- `publish_10hz_loop`: Periodically averages buffered readings and publishes them.

### Packet Format

The service expects packets starting with a header byte `0x55`, followed by a flag byte indicating the data type, an 8-byte payload, and a 1-byte checksum.
