import './App.css'
import { SensorWindow } from './components/SensorWindow.tsx'
import { VisionWindow } from './components/VisionWindow.tsx'

// todo
//  add resize windows
//  add toolbar
//  add way to view depth map switcher
//  add multiple windows
//  add way to view performance of different units on the system
//  add system monitoring metrics - cpu usage, memory usage etc
//  add other components

function App() {
    return (
        <>
            <SensorWindow />
            <VisionWindow />
        </>
    )
}

export default App
