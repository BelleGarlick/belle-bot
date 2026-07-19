import asyncio
import time
import pynmea2
from serial_asyncio import open_serial_connection
from belle_bot.infra.fabric import FabricClient

CLIENT = FabricClient()

PORT = '/dev/ttyUSB0'
BAUDRATE = 115200
FABRIC_ID = "sensors/gps"


class GPSProcessor:
    def __init__(self):
        self.fabric_id = FABRIC_ID

    async def read_loop(self, reader):
        print(f"Starting lightweight GGA-only GPS loop on {PORT}...")
        while True:
            try:
                line_bytes = await reader.readline()
                if not line_bytes:
                    print("No data received from serial port. Reconnecting...")
                    break

                line = line_bytes.decode('ascii', errors='replace').strip()
                if not line.startswith('$') or 'GGA' not in line:
                    continue

                try:
                    msg = pynmea2.parse(line)
                    has_fix = msg.gps_qual > 0

                    payload = {
                        "latitude": float(msg.latitude) if has_fix else 0.0,
                        "longitude": float(msg.longitude) if has_fix else 0.0,
                        "altitude": float(msg.altitude) if (has_fix and msg.altitude) else 0.0,
                        "satellites": int(msg.num_sats) if msg.num_sats else 0,
                        "has_fix": has_fix,
                        "time": time.time()
                    }

                    # Using publish_async for better integration with the async loop
                    await CLIENT.publish_async(self.fabric_id, payload)

                except pynmea2.ParseError:
                    continue
                except Exception as e:
                    print(f"Parsing error: {e}")

            except Exception as e:
                print(f"Serial connection issue: {e}")
                await asyncio.sleep(1)


async def main():
    print(f"Connecting to {PORT}...")
    try:
        reader, writer = await open_serial_connection(url=PORT, baudrate=BAUDRATE)
    except Exception as e:
        print(f"Error opening serial port: {e}")
        return

    processor = GPSProcessor()
    try:
        await processor.read_loop(reader)
    except asyncio.CancelledError:
        pass
    finally:
        writer.close()
        await writer.wait_closed()
        print("GPS serial port closed.")


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Stopping GPS service.")
