import { useEffect, useState, useRef, useMemo, type RefObject } from "react";
import {
    BOTTOM_BAR_HEIGHT,
    DebugText,
    DebugTextContainer,
    ResizeCorner,
    TOPBAR_HEIGHT,
    WindowBottomBar,
    WindowContainer,
    WindowTopBar,
} from "./styles.ts";
import { Switcher, type SwitcherI } from "./Switcher.tsx";
import type { WindowSize } from "./utils.ts";

class ZIndexCount {
    static counter: number = 1;
}

export function RobovizWindow({
    title,
    canvasRef,
    setCanvasSize,
    debugText,
    actions,
}: {
    title: string;
    setCanvasSize: (val: { width: number; height: number }) => void;
    canvasRef: RefObject<HTMLCanvasElement | null>;
    debugText?: string[];
    actions?: SwitcherI[];
}) {
    const [zIndex, setZIndex] = useState(0);

    const [windowPosition, setWindowPosition] = useState({ top: 10, left: 10 });
    const [windowSize, setWindowSize] = useState<WindowSize>({
        width: 500,
        height: 350,
    });
    const [isDragging, setIsDragging] = useState(false);
    const [isResizing, setIsResizing] = useState(false);
    const dragStartRef = useRef({ x: 0, y: 0 });

    useEffect(() => {
        ZIndexCount.counter += 1;
        // eslint-disable-next-line react-hooks/set-state-in-effect
        setZIndex(ZIndexCount.counter);
    }, []);

    const bottomBarHeight = actions ? BOTTOM_BAR_HEIGHT : 0;
    const innerHeight = windowSize.height - TOPBAR_HEIGHT - bottomBarHeight;

    const canvasSize = useMemo(
        () => ({
            width: windowSize.width * 2,
            height: innerHeight * 2,
        }),
        [windowSize.width, windowSize.height, innerHeight],
    );

    useEffect(() => {
        setCanvasSize(canvasSize);
    }, [canvasSize, setCanvasSize]);

    useEffect(() => {
        const handleMouseMove = (e: MouseEvent) => {
            if (isDragging) {
                const deltaX = e.clientX - dragStartRef.current.x;
                const deltaY = e.clientY - dragStartRef.current.y;
                setWindowPosition((prev) => ({
                    top: prev.top + deltaY,
                    left: prev.left + deltaX,
                }));
                dragStartRef.current = { x: e.clientX, y: e.clientY };
            } else if (isResizing) {
                const newWidth = Math.max(
                    200,
                    windowSize.width + (e.clientX - dragStartRef.current.x),
                );
                const newHeight = Math.max(
                    150,
                    windowSize.height + (e.clientY - dragStartRef.current.y),
                );
                setWindowSize({ width: newWidth, height: newHeight });
                dragStartRef.current = { x: e.clientX, y: e.clientY };
            }
        };

        const handleMouseUp = () => {
            setIsDragging(false);
            setIsResizing(false);
        };

        if (isDragging || isResizing) {
            window.addEventListener("mousemove", handleMouseMove);
            window.addEventListener("mouseup", handleMouseUp);
        }

        return () => {
            window.removeEventListener("mousemove", handleMouseMove);
            window.removeEventListener("mouseup", handleMouseUp);
        };
    }, [isDragging, isResizing, windowSize.width, windowSize.height]);

    return (
        <WindowContainer
            style={{
                zIndex,
                top: windowPosition.top,
                left: windowPosition.left,
                width: windowSize.width,
                height: windowSize.height,
            }}
        >
            <WindowTopBar
                onMouseDown={(e) => {
                    if (ZIndexCount.counter != zIndex) {
                        ZIndexCount.counter += 1;
                        setZIndex(ZIndexCount.counter);
                    }
                    setIsDragging(true);
                    dragStartRef.current = { x: e.clientX, y: e.clientY };
                }}
            >
                {title}
            </WindowTopBar>

            <canvas
                ref={canvasRef}
                width={canvasSize.width}
                height={canvasSize.height}
                style={{
                    height: innerHeight,
                    width: windowSize.width,
                    display: "block",
                }}
            />

            <ResizeCorner
                onMouseDown={(e) => {
                    setIsResizing(true);
                    dragStartRef.current = { x: e.clientX, y: e.clientY };
                }}
            />

            {actions && (
                <WindowBottomBar>
                    {actions.map((switcher) => (
                        <Switcher switcher={switcher} />
                    ))}
                </WindowBottomBar>
            )}

            <DebugTextContainer
                style={{
                    bottom: bottomBarHeight + 8,
                }}
            >
                {(debugText ?? []).map((text) => (
                    <DebugText>{text}</DebugText>
                ))}
            </DebugTextContainer>
        </WindowContainer>
    );
}
