import { useEffect, useRef, useState } from 'react'
import { useFabricProvider } from '../Roboviz/contexts/ServerContext.tsx'
import { RobovizWindow } from './RobovizWindow'

export function SensorWindow() {
    const { listen, stopListening } = useFabricProvider()
    const canvasRef = useRef<HTMLCanvasElement | undefined>(undefined)
    const [windowSize, setWindowSize] = useState<{ width: number; height: number }>({
        width: 0,
        height: 0,
    })

    useEffect(() => {
        const listenerId = listen('sensors/camera', (x) => {
            const image = new Image()
            image.onload = function () {
                // todo workout the scale for this cos atm it's stretched over the image

                const canvas = canvasRef.current?.getContext('2d')
                canvas?.drawImage(image, 0, 0, windowSize.width, windowSize.height)
            }
            image.src = `data:image/png;base64,${x['rgb']}`
        })

        return () => stopListening('sensors/camera', listenerId)
    }, [windowSize, listen, stopListening])

    return (
        <RobovizWindow
            title="Sensors"
            debugText={['sensors/camera']}
            setCanvasSize={setWindowSize}
            canvasRef={canvasRef}
        />
    )
}
