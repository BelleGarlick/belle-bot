import serial
import pynmea2
import time

SERIAL_PORT = '/dev/ttyACM0'
BAUD_RATE = 9600  # Change to 38400 if characters become unreadable


def run_gps_service():
    print(f"Connecting to u-blox 7 receiver on {SERIAL_PORT}...")

    try:
        # Open port with a 1-second read timeout
        ser = serial.Serial(SERIAL_PORT, baudrate=BAUD_RATE, timeout=1)

        # Flush buffers to ensure we start on a clean line boundary
        ser.reset_input_buffer()

        while True:
            try:
                # Read line, decode, and clean up padding whitespace
                raw_bytes = ser.readline()
                if not raw_bytes:
                    continue

                line = raw_bytes.decode('ascii', errors='ignore').strip()

                print(line)

                # Only parse lines that look like valid NMEA strings
                if line.startswith('$'):
                    try:
                        msg = pynmea2.parse(line)

                        # Handle GPGGA (Global Positioning Fix Data)
                        if isinstance(msg, pynmea2.GGA):
                            # gps_qual: 0 = invalid, 1 = GPS fix, 2 = DGPS fix
                            if msg.gps_qual > 0:
                                print(
                                    f"[FIX DATA] Lat: {msg.latitude:.6f}° | Lon: {msg.longitude:.6f}° | Sats: {msg.num_sats} | Alt: {msg.altitude}m")
                            else:
                                print(
                                    f"[NO FIX] System Time: {msg.timestamp} UTC | Waiting for satellites... (Sats: {msg.num_sats})")

                        # Handle GPGSV (Satellites in View)
                        elif isinstance(msg, pynmea2.GSV):
                            print(f"[SKY MAP] Satellites physically in view: {msg.num_sv_in_view}")

                        # Handle GPRMC (Recommended Minimum Specific GNSS Data)
                        elif isinstance(msg, pynmea2.RMC):
                            # status: 'A' = Active (Valid), 'V' = Void (Invalid)
                            if msg.status == 'A':
                                print(f"[NAV DATA] Speed: {msg.spd_over_grnd} knots | True Course: {msg.true_course}°")
                            else:
                                # Even without a positional fix, RMC sentences give us the date from satellite clocks
                                print(f"[NO FIX] Syncing Date from Constellation: {msg.datestamp}")

                    except pynmea2.ParseError:
                        # Malformed checksums or partial lines on startup are common; skip them
                        continue

            except serial.SerialException as se:
                print(f"Read error: {se}")
                time.sleep(1)  # Brief cooldown if serial connection stutters

    except KeyboardInterrupt:
        print("\nShutting down GPS service context cleanly.")
    except serial.SerialException as e:
        print(f"Could not connect to port {SERIAL_PORT}: {e}")


if __name__ == "__main__":
    run_gps_service()