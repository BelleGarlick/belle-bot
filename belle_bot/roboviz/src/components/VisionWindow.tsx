import { useEffect, useRef, useState } from 'react'
import { useFabricProvider } from '../contexts/ServerContext.tsx'
import { RobovizWindow } from './RobovizWindow.tsx'

export function VisionWindow() {
    const { listen, stopListening } = useFabricProvider()
    const canvasRef = useRef<HTMLCanvasElement | undefined>(undefined)
    const [windowSize, setWindowSize] = useState<{ width: number; height: number }>({
        width: 0,
        height: 0,
    })

    const [image, setImage] = useState<HTMLImageElement>()
    const [cameraSize, setCameraSize] = useState([0, 0])
    const [boundingBoxes, setBoundingBoxes] = useState<number[][]>([])

    useEffect(() => {
        const listenerId = listen('sensors/camera', (x) => {
            const image = new Image()
            image.onload = function () {
                setImage(image)
                setCameraSize(JSON.parse(x['shape']))
                // todo workout the scale for this cos atm it's stretched over the image
            }
            image.src = `data:image/png;base64,${x['rgb']}`
        })
        const visionBoundingBoxesId = listen('vision/bounding_boxes', (x) => {
            const binaryData = Uint8Array.fromBase64(x.predictions)
            const predictions = new Float32Array(binaryData.buffer)
            const flatArray = new Float32Array(predictions)

            const cols = 6
            const rows = flatArray.length / cols

            const nestedArray = []
            for (let i = 0; i < rows; i++) {
                // Slice extracts 6 elements per row
                nestedArray.push(flatArray.slice(i * cols, (i + 1) * cols))
            }
            setBoundingBoxes(nestedArray)
        })

        return () => {
            stopListening('sensors/camera', listenerId)
            stopListening('vision/bounding_boxes', visionBoundingBoxesId)
        }
    }, [listen, stopListening])

    useEffect(() => {
        if (!image) return

        const ctx = canvasRef.current?.getContext('2d')
        ctx?.drawImage(image, 0, 0, windowSize.width, windowSize.height)

        const scale = windowSize.width / cameraSize[1]

        boundingBoxes
            .filter((x) => x[5] == 0)
            .forEach((x) => {
                ctx.beginPath()
                ctx?.rect(x[0] * scale, x[1] * scale, (x[2] - x[0]) * scale, (x[3] - x[1]) * scale)
                ctx.strokeStyle = `rgb(0, 255, 0)`
                ctx.lineWidth = 5
                ctx.stroke()
            })
    }, [image, windowSize, boundingBoxes, cameraSize])

    return (
        <RobovizWindow
            title="Vision Pipeline"
            debugText="vision/bounding_boxes"
            setCanvasSize={setWindowSize}
            canvasRef={canvasRef}
        />
    )
}
