import { useEffect, useRef, useState } from 'react'
import { RobovizWindow } from '../RobovizWindow'
import { render } from './renderers.ts'

export function ComponentConnectionWindow() {
    const canvasRef = useRef<HTMLCanvasElement | undefined>(undefined)
    const [windowSize, setWindowSize] = useState<{ width: number; height: number }>({
        width: 0,
        height: 0,
    })

    useEffect(() => {
        let rendering = true
        let lasttime = null

        function renderer(d: number) {
            const ctx = canvasRef.current?.getContext('2d')
            if (!ctx) return

            if (lasttime === null) lasttime = d
            render({ time: d, delta: d - lasttime }, windowSize, ctx)
            lasttime = d

            if (rendering)
                requestAnimationFrame(renderer)
        }

        // todo have a way to stop rendering
        renderer(0)

        return () => {
            rendering = false
        }
    }, [windowSize])

    return <RobovizWindow title="Components" setCanvasSize={setWindowSize} canvasRef={canvasRef} />
}
