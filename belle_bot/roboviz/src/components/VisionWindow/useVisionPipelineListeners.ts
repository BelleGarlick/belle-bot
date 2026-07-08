import { useFabricProvider } from '../../Roboviz/contexts/ServerContext.tsx'
import { useEffect, useState } from 'react'

export const BOUNDING_BOX_ID = 'vision/bounding-boxes'
export const POSES_ID = 'vision/pose-estimation'
export const SEGMENTATION_ID = 'vision/segmentation'
export const FACIAL_RECOGNITION_ID = 'vision/facial-recognition'

export function useVisionPipelineListeners(selectedCamera: string) {
    const { listen, stopListening } = useFabricProvider()

    const [image, setImage] = useState<HTMLImageElement>()
    const [cameraSize, setCameraSize] = useState([0, 0])
    const [boundingBoxes, setBoundingBoxes] = useState<Float32Array[]>([])
    const [poses, setPoses] = useState<Float32Array[][]>([])
    const [masks, setMasks] = useState<{
        masks: { class: number; mask: Float32Array }[]
        duration: number
    }>([])
    const [faces, setFaces] = useState<{
        faces: { confidence: number; bbox: Float32Array }[]
        duration: number
    }>([])

    useEffect(() => {
        const listenerId = listen('sensors/camera', (x) => {
            const image = new Image()
            image.onload = function () {
                setImage(image)
                setCameraSize(JSON.parse(x['shape']))
                // todo workout the scale for this cos atm it's stretched over the image
            }
            image.src = `data:image/png;base64,${x[selectedCamera]}`
        })

        const visionBoundingBoxesId = listen('vision/bounding-boxes', (x) => {
            const binaryData = Uint8Array.fromBase64(x.predictions)
            const flatArray = new Float32Array(binaryData.buffer)

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

            setMasks({ masks, duration: x['__duration'] })
        })

        const facialRecognitionListenerId = listen(FACIAL_RECOGNITION_ID, (x) => {
            const detections = JSON.parse(x['faces'])

            setFaces({
                faces: detections.map((det) => ({
                    bbox: new Float32Array(Uint8Array.fromBase64(det['bbox']).buffer),
                    confidence: det.confidence,
                })),
                duration: x['__duration'],
            })
        })

        // todo add face detection

        return () => {
            stopListening('sensors/camera', listenerId)
            stopListening(BOUNDING_BOX_ID, visionBoundingBoxesId)
            stopListening(POSES_ID, poseEstimationListenerId)
            stopListening(POSES_ID, segmentationsListenerId)
            stopListening(FACIAL_RECOGNITION_ID, facialRecognitionListenerId)
        }
        // todo dont do it on listener
    }, [listen, stopListening])

    return {
        image,
        cameraSize,
        boundingBoxes,
        poses,
        masks,
        faces,
    }
}
