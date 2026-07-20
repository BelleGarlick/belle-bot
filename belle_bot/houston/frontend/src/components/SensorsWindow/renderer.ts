import { THEME } from "../../Roboviz/utils.tsx";
import { ODOM_HISTORY, type OdometryItem } from "./utils.ts";
import type { WindowSize } from "../RobovizWindow/utils.ts";

export function renderCameraImage(
    ctx: CanvasRenderingContext2D,
    windowSize: WindowSize,
    data,
    selectedCamera: "rgb" | "depth",
) {
    const image = new Image();
    image.onload = function () {
        // Clear the canvas from previous frames
        ctx.clearRect(0, 0, windowSize.width, windowSize.height);

        // --- Aspect Ratio Math (Contain Logic) ---
        const imageRatio = image.width / image.height;
        const windowRatio = windowSize.width / windowSize.height;

        let drawWidth = windowSize.width;
        let drawHeight = windowSize.height;
        let offsetX = 0;
        let offsetY = 0;

        if (imageRatio > windowRatio) {
            // Image is wider than the window container
            drawHeight = windowSize.width / imageRatio;
            offsetY = (windowSize.height - drawHeight) / 2;
        } else {
            // Image is taller than the window container
            drawWidth = windowSize.height * imageRatio;
            offsetX = (windowSize.width - drawWidth) / 2;
        }

        ctx.drawImage(image, offsetX, offsetY, drawWidth, drawHeight);
    };

    image.src = `data:image/jpg;base64,${data[selectedCamera]}`;
}

export function renderOdometryData(
    ctx: CanvasRenderingContext2D,
    windowSize: WindowSize,
    selectedImu: "acc" | "gyro" | "angle",
    odometryHistory: OdometryItem[],
) {
    ctx.clearRect(0, 0, windowSize.width, windowSize.height);

    ctx.strokeStyle = THEME;
    ctx.lineWidth = 3;

    let scale = windowSize.height / 2;
    if (selectedImu === "acc") scale *= 0.2;
    if (selectedImu === "gyro") scale *= 0.001;
    if (selectedImu === "angle") scale *= 1 / 240;

    // Render all axis of the selected sensor item
    for (let axis = 0; axis < 3; axis++) {
        ctx.beginPath();
        for (let i = 0; i < odometryHistory.length; i++) {
            if (i >= odometryHistory.length) continue;

            const item = odometryHistory[i];

            const x = windowSize.width * (i / (ODOM_HISTORY - 1));
            const y = item[selectedImu][axis] * scale + windowSize.height / 2;

            if (i == 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        }
        ctx.stroke();
    }
}
