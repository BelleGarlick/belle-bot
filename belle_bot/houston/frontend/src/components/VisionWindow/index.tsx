import { useEffect, useRef, useState } from "react";
import { RobovizWindow } from "../RobovizWindow";
import {
    BOUNDING_BOX_ID,
    FACIAL_RECOGNITION_ID,
    POSES_ID,
    SEGMENTATION_ID,
    useVisionPipelineListeners,
} from "./useVisionPipelineListeners.ts";

// todo add depth
//  add connections between the pose nodes
//  sync up the video a little smarter
//  add message if the data isn't available
//  add compute time per prediction
//  update the renderer to improve aspect ratio stuff
//  split up the render function into diff functions within a file
//  add tests for this window
//  say how many frames behind the current frame is
//  inject a time that the fabric is reporting items from so that it can be used for time keeping so we can crop interesting events perhaps this happens in a diff tool

export function VisionWindow() {
    const canvasRef = useRef<HTMLCanvasElement | undefined>(undefined);
    const [windowSize, setWindowSize] = useState<{
        width: number;
        height: number;
    }>({
        width: 0,
        height: 0,
    });

    // todo if no data then render a message saying that

    const [selectedCamera, setSelectedCamera] = useState("rgb");
    const [selectedModes, setSelectedModes] = useState([SEGMENTATION_ID]);

    const { image, cameraSize, boundingBoxes, poses, masks, faces } =
        useVisionPipelineListeners(selectedCamera);

    // todo incorperate the frame pairing when rendering the data

    function renderer() {
        if (!image) return;

        const ctx = canvasRef.current?.getContext("2d");
        ctx?.drawImage(image, 0, 0, windowSize.width, windowSize.height);

        const scale = windowSize.width / cameraSize[1];

        if (selectedModes.includes(BOUNDING_BOX_ID)) {
            renderBoundingBoxes();
            boundingBoxes
                .filter((x) => x[5] == 0)
                .forEach((x) => {
                    ctx!.beginPath();
                    ctx!.rect(
                        x[0] * scale,
                        x[1] * scale,
                        (x[2] - x[0]) * scale,
                        (x[3] - x[1]) * scale,
                    );
                    ctx!.strokeStyle = `rgb(0, 255, 0)`;
                    ctx!.lineWidth = 5;
                    ctx!.stroke();
                });
        }

        if (selectedModes.includes(POSES_ID)) {
            poses.forEach((person) => {
                person.forEach((keyPoint) => {
                    if (keyPoint[2] < 0.5) return;

                    ctx!.beginPath();
                    ctx!.arc(
                        keyPoint[0] * scale,
                        keyPoint[1] * scale,
                        7,
                        0,
                        2 * Math.PI,
                    );
                    ctx!.fillStyle = `rgb(0, 255, 0)`;
                    ctx!.fill();
                });
            });
        }

        if (selectedModes.includes(SEGMENTATION_ID)) {
            masks.masks?.forEach((x) => {
                if (x.class !== 0) return;

                ctx!.beginPath();
                ctx!.fillStyle = `rgb(0, 255, 0, 0.3)`;
                for (let i = 0; i < x.mask.length; i += 2) {
                    ctx!.lineTo(x.mask[i] * scale, x.mask[i + 1] * scale);
                }
                ctx!.fill();
            });
        }

        if (selectedModes.includes(FACIAL_RECOGNITION_ID)) {
            faces.faces?.forEach((x) => {
                ctx!.beginPath();
                ctx!.rect(
                    x.bbox[0] * scale,
                    x.bbox[1] * scale,
                    (x.bbox[2] - x.bbox[0]) * scale,
                    (x.bbox[3] - x.bbox[1]) * scale,
                );
                ctx!.strokeStyle = `rgb(0, 255, 0)`;
                ctx!.lineWidth = 5;
                ctx!.stroke();
            });
        }

        requestAnimationFrame(renderer);
    }

    useEffect(() => {
        // todo have a way to stop rendering
        renderer();
    }, []);

    const debugText = [selectedModes[0]];
    if (selectedModes[0] === SEGMENTATION_ID && masks.duration) {
        debugText.push(`Duration: ${Math.round(masks.duration)}ms`);
    }
    if (selectedModes[0] === FACIAL_RECOGNITION_ID && faces.duration) {
        debugText.push(`Duration: ${Math.round(faces.duration)}ms`);
    }

    return (
        <RobovizWindow
            title="Vision"
            debugText={debugText}
            setCanvasSize={setWindowSize}
            canvasRef={canvasRef}
            actions={[
                {
                    selection: selectedModes,
                    options: [
                        { text: "BB", value: BOUNDING_BOX_ID },
                        { text: "Pose", value: POSES_ID },
                        { text: "Seg", value: SEGMENTATION_ID },
                        { text: "Faces", value: FACIAL_RECOGNITION_ID },
                    ],
                    setSelection: setSelectedModes,
                },
                {
                    selection: [selectedCamera],
                    options: [
                        { text: "RGB", value: "rgb" },
                        { text: "Depth", value: "depth" },
                    ],
                    setSelection: (x) => setSelectedCamera(x[0]),
                },
            ]}
        />
    );
}
