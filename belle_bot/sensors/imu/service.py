import asyncio
import time

import numpy as np
import struct
from serial_asyncio import open_serial_connection

from belle_bot.fabric import FabricClient

PORT = '/dev/cu.usbserial-130'
BAUDRATE = 9600
HZ_10_INTERVAL = 0.1  # 10hz rate
FABRIC_SERVICE_ID = "sensors/imu"
CLIENT = FabricClient()


def decode_acceleration(data: bytes):
    """
    Decodes acceleration data from the serial port.

    Ax, Ay, Az are signed 16-bit integers (little endian) which we convert into
    standard units (g)

    :param data: The raw acceleration data.
    :return: The decoded acceleration data in the x, y, z format
    """
    ax, ay, az, temp = struct.unpack('<hhhh', data[:8])
    arr = np.array([ax, ay, az])

    # convert to standard units
    return arr / 32768.0 * 16.0


def decode_angular_velocity(data):
    """
    Decodes angular velocity data from the serial port.

    :param data: The raw angular velocity data.
    :return: The decoded angular velocity data in the x, y, z format
    """
    # Wx, Wy, Wz are signed 16-bit integers (little endian)
    wx, wy, wz, temp = struct.unpack('<hhhh', data[:8])
    arr = np.array([wx, wy, wz])

    # Convert to standard units (°/s)
    return arr / 32768.0 * 2000.0



def decode_angle(data):
    """
    Decodes angle data from the serial port.

    :param data: The raw angle data.
    :return: The decoded angle data in the x, y, z format
    """
    # Roll, Pitch, Yaw are signed 16-bit integers (little endian)
    roll, pitch, yaw, temp = struct.unpack('<hhhh', data[:8])
    arr = np.array([roll, pitch, yaw])

    # Convert to standard units (°)
    return arr / 32768.0 * 180.0


async def read_exact(reader, n: int) -> bytes:
    """Helper to read exactly n bytes asynchronously, avoiding readexact limitations."""
    data = b''
    while len(data) < n:
        chunk = await reader.read(n - len(data))
        if not chunk:
            raise asyncio.IncompleteReadError(data, n)
        data += chunk
    return data


class IMUProcessor:
    """
    Handles reading, decoding, and periodic publishing of IMU data.

    This class maintains buffers for acceleration and angular velocity readings,
    averages them at a fixed interval (10Hz), and publishes the results.
    """
    def __init__(self):
        """Initializes the IMUProcessor with empty buffers and default angle."""
        self.accel_buffer = []
        self.gyro_buffer = []
        self.latest_angle = np.array([0.0, 0.0, 0.0])

    async def read_loop(self, reader):
        """
        Continuously reads and decodes the raw IMU serial stream.

        This method parses incoming bytes from the serial reader, looking for
        specific packet headers and flags to identify acceleration (0x51),
        angular velocity (0x52), and angle (0x53) data. Decoded data is
        stored in internal buffers or as the latest angle reading.

        :param reader: The stream reader for the serial connection.
        """
        while True:
            try:
                # Look for the header byte (0x55)
                head = await read_exact(reader, 1)
                if head == b'\x55':
                    flag = await read_exact(reader, 1)

                    # Read 8 bytes payload + 1 byte checksum
                    packet = await read_exact(reader, 9)
                    payload = packet[:8]
                    checksum = packet[8]

                    # Checksum validation
                    calculated_checksum = (0x55 + flag[0] + sum(payload)) & 0xFF
                    if calculated_checksum != checksum:
                        continue

                    # Decode and append to window buffer
                    if flag == b'\x51':
                        self.accel_buffer.append(decode_acceleration(payload))
                    elif flag == b'\x52':
                        self.gyro_buffer.append(decode_angular_velocity(payload))
                    elif flag == b'\x53':
                        self.latest_angle = decode_angle(payload)

            except asyncio.IncompleteReadError:
                print("\nIncomplete read from serial port. Reconnecting...")
                await asyncio.sleep(1)
            except Exception as e:
                print(f"\nError in read loop: {e}")
                await asyncio.sleep(1)

    async def publish_10hz_loop(self):
        """
        Wakes up every 100ms to aggregate, average, and publish the data.

        This method runs indefinitely, calculating the mean of all acceleration
        and gyro readings collected in the last 100ms interval. The averaged
        data, along with the latest absolute angle, is published to the
        configured fabric service.
        """
        print("Starting 10Hz data collation...")

        next_time = asyncio.get_event_loop().time()

        while True:
            next_time += HZ_10_INTERVAL
            await asyncio.sleep(max(0, next_time - asyncio.get_event_loop().time()))

            accels = self.accel_buffer
            gyros = self.gyro_buffer
            self.accel_buffer = []
            self.gyro_buffer = []

            # Compute averages using numpy along the sample axis (axis 0)
            avg_acc = np.mean(accels, axis=0) if accels else np.array([0.0, 0.0, 0.0])
            avg_gyro = np.mean(gyros, axis=0) if gyros else np.array([0.0, 0.0, 0.0])
            angle = self.latest_angle

            CLIENT.publish(FABRIC_SERVICE_ID, {
                "time": np.array(time.time()).astype(np.float32),
                "acc": avg_acc.astype(np.float32),
                "gyro": avg_gyro.astype(np.float32),
                "angle": angle.astype(np.float32)
            })


async def main():
    """
    Main entry point for the IMU service.

    Opens the serial connection, initializes the IMUProcessor, and starts
    the concurrent read and publish loops.
    """
    print(f"Opening connection to {PORT} at {BAUDRATE} baud...")
    try:
        reader, writer = await open_serial_connection(url=PORT, baudrate=BAUDRATE)
    except Exception as e:
        print(f"Error opening serial port: {e}")
        return

    processor = IMUProcessor()

    read_task = asyncio.create_task(processor.read_loop(reader))
    report_task = asyncio.create_task(processor.publish_10hz_loop())

    try:
        await asyncio.gather(read_task, report_task)
    except asyncio.CancelledError:
        pass
    finally:
        writer.close()
        await writer.wait_closed()
        print("\nSerial port closed.")


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopping data collection.")
