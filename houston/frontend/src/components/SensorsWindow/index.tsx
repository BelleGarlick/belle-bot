import { useEffect, useRef, useState } from "react";
import { useFabricProvider } from "../../Roboviz/contexts/ServerContext.tsx";
import { RobovizWindow } from "../RobovizWindow";
import { renderCameraImage, renderOdometryData } from "./renderer.ts";
import { fromNumpyF32 } from "../../Roboviz/utils.tsx";
import { ODOM_HISTORY, type OdometryItem, Sensor } from "./utils.ts";
import type { SwitcherI } from "../RobovizWindow/Switcher.tsx";

export function SensorWindow() {
    const { listen, stopListening } = useFabricProvider();
    const canvasRef = useRef<HTMLCanvasElement | null>(null); // Fixed typings from undefined to null for standard React canvas refs
    const odometryHistory = useRef<OdometryItem[]>([]);
    const [windowSize, setWindowSize] = useState<{
        width: number;
        height: number;
    }>({
        width: 0,
        height: 0,
    });

    const [selectedSensor, setSelectedSensor] = useState<string[]>([
        Sensor.CAMERA,
    ]);

    const [selectedCamera, setSelectedCamera] = useState<"rgb" | "depth">(
        "rgb",
    );
    const [selectedImu, setSelectedImu] = useState<"acc" | "gyro" | "angle">(
        "acc",
    );

    useEffect(() => {
        const cameraListenerId = listen(Sensor.CAMERA, (x) => {
            if (selectedSensor[0] !== Sensor.CAMERA) return;

            const canvasElement = canvasRef.current;
            if (!canvasElement) return;

            const ctx = canvasElement.getContext("2d");
            if (!ctx) return;

            renderCameraImage(ctx, windowSize, x, selectedCamera);
        });

        const imuListenerId = listen(Sensor.IMU, (x) => {
            // Log the data
            odometryHistory.current.push({
                acc: fromNumpyF32(x.acc as string),
                gyro: fromNumpyF32(x.gyro as string),
                angle: fromNumpyF32(x.angle as string),
            });

            const itemsCount = odometryHistory.current.length;
            if (itemsCount > ODOM_HISTORY)
                odometryHistory.current = odometryHistory.current.slice(
                    1,
                    itemsCount,
                );

            // Check whether to render the data or not
            if (selectedSensor[0] !== Sensor.IMU) return;

            const canvasElement = canvasRef.current;
            if (!canvasElement) return;

            const ctx = canvasElement.getContext("2d");
            if (!ctx) return;

            // Render the odom data
            renderOdometryData(
                ctx,
                windowSize,
                selectedImu,
                odometryHistory.current,
            );
        });

        return () => {
            stopListening(Sensor.CAMERA, cameraListenerId);
            stopListening(Sensor.IMU, imuListenerId);
        };
    }, [
        windowSize,
        listen,
        stopListening,
        selectedCamera,
        selectedSensor,
        selectedImu,
    ]);

    const actionOptions: SwitcherI[] = [
        {
            selection: selectedSensor,
            options: [
                { text: "Camera", value: Sensor.CAMERA },
                { text: "IMU", value: Sensor.IMU },
            ],
            setSelection: setSelectedSensor,
        },
    ];

    if (selectedSensor[0] === Sensor.CAMERA) {
        actionOptions.push({
            selection: [selectedCamera],
            options: [
                { text: "RGB", value: "rgb" },
                { text: "Depth", value: "depth" },
            ],
            setSelection: (x) => setSelectedCamera(x[0] as "rgb" | "depth"),
        });
    }

    if (selectedSensor[0] === Sensor.IMU) {
        actionOptions.push({
            selection: [selectedImu],
            options: [
                { text: "Acc", value: "acc" },
                { text: "Gyro", value: "gyro" },
                { text: "Angle", value: "angle" },
            ],
            setSelection: (x) =>
                setSelectedImu(x[0] as "acc" | "gyro" | "angle"),
        });
    }

    return (
        <RobovizWindow
            title="Sensors"
            debugText={selectedSensor}
            setCanvasSize={setWindowSize}
            canvasRef={canvasRef}
            actions={actionOptions}
        />
    );
}
