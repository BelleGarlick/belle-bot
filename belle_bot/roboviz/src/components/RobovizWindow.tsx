import { useEffect, useState, useRef, useMemo, type RefObject } from 'react'

class ZIndexCount {
    static counter: number = -1000
}

const THEME = '#aa33ff'

export function RobovizWindow({
    title,
    canvasRef,
    setCanvasSize,
    debugText,
    switcher,
}: {
    title: string
    setCanvasSize: (val: { width: number; height: number }) => void
    canvasRef: RefObject<HTMLCanvasElement | undefined>
    debugText: string
    switcher?: {
        selection: string[]
        options: { text: string; value: string }[]
        setSelection: (selection: string[]) => void
    }
}) {
    const [zIndex, setZIndex] = useState(0)

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
        ZIndexCount.counter += 1
        // eslint-disable-next-line react-hooks/set-state-in-effect
        setZIndex(ZIndexCount.counter)
    }, [])

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
                zIndex,
                position: 'absolute',
                top: windowPosition.top,
                left: windowPosition.left,
                width: 500,
                height: 310,
                border: '2px solid #440088',
                color: THEME,
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
                    backgroundColor: '#222',
                    userSelect: 'none',
                    flexDirection: 'row',
                    display: 'flex',
                    gap: 16,
                }}
                onMouseDown={(e) => {
                    if (ZIndexCount.counter != zIndex) {
                        ZIndexCount.counter += 1
                        setZIndex(ZIndexCount.counter)
                    }
                    setIsDragging(true)
                    dragStartRef.current = { x: e.clientX, y: e.clientY }
                }}
            >
                {switcher && (
                    <div
                        style={{
                            display: 'flex',
                            flexDirection: 'row',
                            borderRadius: '4px',
                            overflow: 'hidden',
                        }}
                    >
                        {switcher.options.map((option) => {
                            return (
                                <div
                                    style={{
                                        backgroundColor: switcher.selection.includes(option.value)
                                            ? THEME
                                            : 'black',
                                        color: switcher.selection.includes(option.value)
                                            ? 'black'
                                            : THEME,
                                        padding: '0px 8px',
                                    }}
                                    onClick={() => {
                                        switcher.setSelection([option.value])
                                    }}
                                >
                                    {option.text}
                                </div>
                            )
                        })}
                    </div>
                )}

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
