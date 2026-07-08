import { SensorWindow } from '../components/SensorWindow.tsx'
import { VisionWindow } from '../components/VisionWindow'
import { type CSSProperties, type ReactNode, useState } from 'react'
import styled from '@emotion/styled'
import { THEME } from './utils.tsx'

// todo
//  add toolbar
//  add way to view performance of different units on the system
//  add system monitoring metrics - cpu usage, memory usage etc
//  add other components

const ToolbarButton = styled.button`
    height: 80px;
    border: 0px;
    border-radius: 0px;
    background-color: #222222;
    color: ${THEME};
    font-weight: bold;
    font-family: monospace;
`

export function RoboViz({ style }: { style?: CSSProperties }) {
    const [windows, setWindows] = useState<ReactNode[]>([])

    return (
        <div
            style={{
                ...style,
                flexDirection: 'row',
                display: 'grid',
                flexGrow: 1,
                gridTemplateColumns: '100px auto',
            }}
        >
            <div
                style={{
                    height: '100%',
                    flexDirection: 'column',
                    display: 'flex',
                    borderRight: '2px solid #112',
                }}
            >
                <ToolbarButton
                    onClick={() => {
                        setWindows([...windows, <SensorWindow />])
                    }}
                >
                    Sensors
                </ToolbarButton>
                <ToolbarButton
                    onClick={() => {
                        setWindows([...windows, <VisionWindow />])
                    }}
                >
                    Vision
                </ToolbarButton>
            </div>
            <div
                style={{
                    position: 'relative',
                    overflow: 'hidden',
                }}
            >
                {windows}
            </div>
        </div>
    )
}
