import { useEffect, useRef, useState } from 'react'
import { useFabricProvider } from '../contexts/ServerContext.tsx'
import { RobovizWindow } from './RobovizWindow.tsx'

const BOUNDING_BOX_ID = 'vision/bounding-boxes'
const POSES_ID = 'vision/pose-estimation'
const SEGMENTATION_ID = 'vision/segmentation'

// todo add depth
// todo add connections between the pose nodes
// todo sync up the video a little smarter
// todo add message if the data isn't available
// todo may want to pull the boxes out of pose / seg masks
// todo add facial detection

export function VisionWindow() {
    const { listen, stopListening } = useFabricProvider()
    const canvasRef = useRef<HTMLCanvasElement | undefined>(undefined)
    const [windowSize, setWindowSize] = useState<{ width: number; height: number }>({
        width: 0,
        height: 0,
    })

    // todo if no data then render a message saying that

    const [selectedModes, setSelectedModes] = useState([SEGMENTATION_ID])
    const [image, setImage] = useState<HTMLImageElement>()
    const [cameraSize, setCameraSize] = useState([0, 0])
    const [boundingBoxes, setBoundingBoxes] = useState<Float32Array[]>([])
    const [poses, setPoses] = useState<Float32Array[][]>([])
    const [masks, setMasks] = useState<{ class: number; mask: Float32Array }[]>([])

    // todo incorperate the frame pairing when rendering the data

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

        const visionBoundingBoxesId = listen('vision/bounding-boxes', (x) => {
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

        const poseEstimationListenerId = listen(POSES_ID, (x) => {
            const binaryData = Uint8Array.fromBase64(x.predictions)
            const predictions = new Float32Array(binaryData.buffer)
            const flatArray = new Float32Array(predictions)

            const peoplePoses: Float32Array[][] = []
            for (let i = 0; i < flatArray.length; i += 51) {
                const peopleKeyPoints = []
                const peopleData = flatArray.slice(i, i + 51)

                for (let p = 0; p < 51; p += 3) {
                    peopleKeyPoints.push(peopleData.slice(p, p + 3))
                }

                peoplePoses.push(peopleKeyPoints)
            }
            setPoses(peoplePoses)
        })

        const segmentationsListenerId = listen(SEGMENTATION_ID, (x) => {
            const classes = JSON.parse(x['classes'])
            const maskLengths = JSON.parse(x['mask-lengths'])

            const binaryData = Uint8Array.fromBase64(x['masks'])
            const predictions = new Float32Array(binaryData.buffer)
            const flatArray = new Float32Array(predictions)

            const masks: { class: number; mask: Float32Array }[] = []

            let mIdx = 0
            for (let mask = 0; mask < maskLengths.length; mask += 1) {
                const start = mIdx
                const end = mIdx + maskLengths[mask]
                mIdx += maskLengths[mask]

                masks.push({
                    class: classes[mask],
                    mask: flatArray.slice(start * 2, end * 2),
                })
            }

            setMasks(masks)
        })

        // todo add face detection

        return () => {
            stopListening('sensors/camera', listenerId)
            stopListening(BOUNDING_BOX_ID, visionBoundingBoxesId)
            stopListening(POSES_ID, poseEstimationListenerId)
            stopListening(POSES_ID, segmentationsListenerId)
        }
    }, [listen, stopListening])

    useEffect(() => {
        if (!image) return

        const ctx = canvasRef.current?.getContext('2d')
        ctx?.drawImage(image, 0, 0, windowSize.width, windowSize.height)

        const scale = windowSize.width / cameraSize[1]

        if (selectedModes.includes(BOUNDING_BOX_ID)) {
            boundingBoxes
                .filter((x) => x[5] == 0)
                .forEach((x) => {
                    ctx!.beginPath()
                    ctx!.rect(
                        x[0] * scale,
                        x[1] * scale,
                        (x[2] - x[0]) * scale,
                        (x[3] - x[1]) * scale,
                    )
                    ctx!.strokeStyle = `rgb(0, 255, 0)`
                    ctx!.lineWidth = 5
                    ctx!.stroke()
                })
        }

        if (selectedModes.includes(POSES_ID)) {
            poses.forEach((person) => {
                person.forEach((keyPoint) => {
                    if (keyPoint[2] < 0.5) return

                    ctx!.beginPath()
                    ctx!.arc(keyPoint[0] * scale, keyPoint[1] * scale, 7, 0, 2 * Math.PI)
                    ctx!.fillStyle = `rgb(0, 255, 0)`
                    ctx!.fill()
                })
            })
        }

        if (selectedModes.includes(SEGMENTATION_ID)) {
            masks.forEach((x) => {
                if (x.class !== 0)
                    return;

                ctx!.beginPath()
                ctx!.fillStyle = `rgb(0, 255, 0, 0.3)`
                for (let i = 0; i < x.mask.length; i += 2) {
                    ctx!.lineTo(x.mask[i] * scale, x.mask[i + 1] * scale, 7, 0, 2 * Math.PI)
                }
                ctx!.fill()
            })
        }
    }, [image, windowSize, boundingBoxes, poses, cameraSize])

    return (
        <RobovizWindow
            title="Vision Pipeline"
            debugText={selectedModes[0]}
            setCanvasSize={setWindowSize}
            canvasRef={canvasRef}
            switcher={{
                selection: selectedModes,
                options: [
                    { text: 'BB', value: BOUNDING_BOX_ID },
                    { text: 'Pose', value: POSES_ID },
                    { text: 'Seg', value: SEGMENTATION_ID },
                ],
                setSelection: setSelectedModes,
            }}
        />
    )
}
