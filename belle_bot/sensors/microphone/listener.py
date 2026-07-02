import sounddevice as sd
import numpy as np
import base64
from belle_bot.fabric import FabricClient
from belle_bot.sensors.microphone import config

# Initialize the output stream
# We use a 'non-blocking' output stream
output_stream = sd.OutputStream(samplerate=16000, channels=1, dtype='int16')
output_stream.start()


def on_audio_message(data):
    audio_bytes = base64.b64decode(data.get("audio"))
    audio_data = np.frombuffer(audio_bytes, dtype='int16')
    output_stream.write(audio_data)


if __name__ == "__main__":
    CLIENT = FabricClient()
    CLIENT.listen(config.FABRIC_ID, callback=on_audio_message)

    while True:
        pass
