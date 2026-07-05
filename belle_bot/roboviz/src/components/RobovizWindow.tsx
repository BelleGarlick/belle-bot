import { useEffect, useState, useRef, useMemo, type RefObject } from 'react'

class ZIndexCount {
    static counter: number = 0
}

export function RobovizWindow({
    title,
    canvasRef,
    setCanvasSize,
    debugText,
}: {
    title: string
    setCanvasSize: (val: { width: number; height: number }) => void
    canvasRef: RefObject<HTMLCanvasElement | undefined>
    debugText: string
}) {
    const [windowPosition, setWindowPosition] = useState({ top: 10, left: 10 })
    const windowSize = { width: 500, height: 350 }
    const [isDragging, setIsDragging] = useState(false)
    const dragStartRef = useRef({ x: 0, y: 0 })

    const topbarHeight = 70
    const canvasSize = useMemo(
        () => ({
            width: windowSize.width * 2,
            height: (windowSize.height - topbarHeight) * 2,
        }),
        [windowSize.width, windowSize.height, topbarHeight],
    )

    useEffect(() => {
        setCanvasSize(canvasSize)
    }, [canvasSize, setCanvasSize])

    useEffect(() => {
        // We define these outside so we can add/remove them reliably
        const handleMouseMove = (e: MouseEvent) => {
            if (!isDragging) return

            const deltaX = e.clientX - dragStartRef.current.x
            const deltaY = e.clientY - dragStartRef.current.y

            setWindowPosition((prev) => ({
                top: prev.top + deltaY,
                left: prev.left + deltaX,
            }))

            dragStartRef.current = { x: e.clientX, y: e.clientY }
        }

        const handleMouseUp = () => {
            setIsDragging(false)
        }

        if (isDragging) {
            window.addEventListener('mousemove', handleMouseMove)
            window.addEventListener('mouseup', handleMouseUp)
        }

        return () => {
            window.removeEventListener('mousemove', handleMouseMove)
            window.removeEventListener('mouseup', handleMouseUp)
        }
    }, [isDragging])

    return (
        <div
            style={{
                position: 'absolute',
                top: windowPosition.top,
                left: windowPosition.left,
                width: 500,
                height: 310,
                border: '2px solid #110022',
                color: '#aa33ff',
                overflow: 'hidden',
                backgroundColor: '#222',
                borderRadius: 16,
                boxShadow: '0px 0px 20px -10px black',
                userSelect: 'none', // Prevents text selection while dragging
            }}
        >
            <div
                style={{
                    textAlign: 'left',
                    padding: '4px 8px',
                    fontWeight: 'bold',
                    fontFamily: 'monospace',
                    fontSize: 'small',
                    cursor: 'grab',
                    backgroundColor: '#333',
                    userSelect: 'none',
                }}
                onMouseDown={(e) => {
                    setIsDragging(true)
                    dragStartRef.current = { x: e.clientX, y: e.clientY }
                }}
            >
                {title}
            </div>

            <canvas
                ref={canvasRef}
                width={canvasSize.width}
                height={canvasSize.height}
                style={{
                    height: windowSize.height - topbarHeight,
                    width: windowSize.width,
                }}
            />

            {debugText && (
                <div
                    style={{
                        position: 'absolute',
                        bottom: '8px',
                        left: '8px',
                        backgroundColor: '#222',
                        borderRadius: 8,
                        padding: '0px 8px',
                        fontFamily: 'monospace',
                        fontSize: 12,
                        boxShadow: '0px 0px 20px -10px black',
                        fontWeight: 'bold',
                    }}
                >
                    {debugText}
                </div>
            )}
        </div>
    )
}
