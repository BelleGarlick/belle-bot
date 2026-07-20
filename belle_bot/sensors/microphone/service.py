import base64
import uuid

import sounddevice as sd
import asyncio
import numpy as np

from belle_bot.infra.fabric import FabricClient

CLIENT = FabricClient()

FABRIC_ID = "sensors/microphone"
SAMPLE_RATE = 16000
BUFFER_SIZE = 8000  # each packet is 0.5s audio

audio_buffer = np.array([], dtype='int16')


async def stream_audio():
    global audio_buffer

    def callback(indata, frames, time, status):
        global audio_buffer
        # Append incoming chunk to our buffer
        audio_buffer = np.append(audio_buffer, indata.flatten())

        # Once we have at least 1 second of audio (16,000 samples)
        if len(audio_buffer) >= BUFFER_SIZE:
            # Take the first 1 second
            to_send = audio_buffer[:BUFFER_SIZE]
            # Keep the remainder (if any) for the next cycle
            audio_buffer = audio_buffer[BUFFER_SIZE:]

            # Encode and publish
            audio_payload = base64.b64encode(to_send).decode('utf-8')

            coro = CLIENT.publish_async(FABRIC_ID, {
                "service_name": FABRIC_ID,
                "audio_id": str(uuid.uuid4()),
                "audio": audio_payload,
                "sample_rate": str(SAMPLE_RATE),
                "channels": str(1)
            })

            asyncio.run_coroutine_threadsafe(coro, loop)

    loop = asyncio.get_running_loop()
    with sd.InputStream(callback=callback, channels=1, samplerate=SAMPLE_RATE, dtype='int16'):
        await asyncio.Future()


if __name__ == '__main__':
    asyncio.run(stream_audio())
