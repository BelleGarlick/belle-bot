import './App.css'
import { SensorWindow } from './components/SensorWindow'
import { VisionWindow } from './components/VisionWindow'
import { ComponentConnectionWindow } from './components/ComponentConnectionWindow'

// todo
//  add toolbar
//  add way to view performance of different units on the system
//  add system monitoring metrics - cpu usage, memory usage etc
//  add other components

function App() {
    return (
        <>
            <SensorWindow />
            <VisionWindow />
            {/*<VisionWindow />*/}

            {/*<ComponentConnectionWindow />*/}
        </>
    )
}

export default App
