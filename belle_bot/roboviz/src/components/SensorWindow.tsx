import { useEffect, useRef, useState } from 'react'
import { useFabricProvider } from '../contexts/ServerContext.tsx'
import { RobovizWindow } from './RobovizWindow'

export function SensorWindow() {
    const { listen, stopListening } = useFabricProvider()
    const canvasRef = useRef<HTMLCanvasElement | null>(null) // Fixed typings from undefined to null for standard React canvas refs
    const [windowSize, setWindowSize] = useState<{ width: number; height: number }>({
        width: 0,
        height: 0,
    })
    const [selectedCamera, setSelectedCamera] = useState('rgb')

    useEffect(() => {
        const listenerId = listen('sensors/camera', (x) => {
            const image = new Image()
            image.onload = function () {
                const canvasElement = canvasRef.current
                if (!canvasElement) return

                const ctx = canvasElement.getContext('2d')
                if (!ctx) return

                // Clear the canvas from previous frames
                ctx.clearRect(0, 0, windowSize.width, windowSize.height)

                // --- Aspect Ratio Math (Contain Logic) ---
                const imageRatio = image.width / image.height
                const windowRatio = windowSize.width / windowSize.height

                let drawWidth = windowSize.width
                let drawHeight = windowSize.height
                let offsetX = 0
                let offsetY = 0

                if (imageRatio > windowRatio) {
                    // Image is wider than the window container
                    drawHeight = windowSize.width / imageRatio
                    offsetY = (windowSize.height - drawHeight) / 2
                } else {
                    // Image is taller than the window container
                    drawWidth = windowSize.height * imageRatio
                    offsetX = (windowSize.width - drawWidth) / 2
                }

                ctx.drawImage(image, offsetX, offsetY, drawWidth, drawHeight)
            }

            // Fixed: Dynamically switch or default to image/png since you are cv2 encoding to .png
            image.src = `data:image/png;base64,${x[selectedCamera]}`
        })

        return () => stopListening('sensors/camera', listenerId)
    }, [windowSize, listen, stopListening, selectedCamera])

    return (
        <RobovizWindow
            title="Sensors"
            debugText={['sensors/camera']}
            setCanvasSize={setWindowSize}
            canvasRef={canvasRef}
            actions={[
                {
                    selection: [selectedCamera],
                    options: [
                        { text: 'RGB', value: 'rgb' },
                        { text: 'Depth', value: 'depth' },
                    ],
                    setSelection: (x) => setSelectedCamera(x[0]),
                },
            ]}
        />
    )
}