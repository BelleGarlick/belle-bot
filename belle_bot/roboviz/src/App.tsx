import {type PropsWithChildren, useEffect, useState, useRef} from 'react'
import './App.css'
import {useFabricProvider} from "./contexts/ServerContext.tsx";

function RobovizWindow({ title, children }: { title: string } & PropsWithChildren) {
  const [windowPosition, setWindowPosition] = useState({ top: 10, left: 10 });
  const [isDragging, setIsDragging] = useState(false);
  const dragStartRef = useRef({ x: 0, y: 0 });

  useEffect(() => {
    // We define these outside so we can add/remove them reliably
    const handleMouseMove = (e: MouseEvent) => {
      if (!isDragging) return;

      const deltaX = e.clientX - dragStartRef.current.x;
      const deltaY = e.clientY - dragStartRef.current.y;

      setWindowPosition((prev) => ({
        top: prev.top + deltaY,
        left: prev.left + deltaX,
      }));

      dragStartRef.current = { x: e.clientX, y: e.clientY };
    };

    const handleMouseUp = () => {
      setIsDragging(false);
    };

    if (isDragging) {
      window.addEventListener('mousemove', handleMouseMove);
      window.addEventListener('mouseup', handleMouseUp);
    }

    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
    };
  }, [isDragging]);

  return (
    <div style={{
      position: 'absolute',
      top: windowPosition.top,
      left: windowPosition.left,
      width: 500,
      height: 310,
      border: '2px solid black',
      overflow: 'hidden',
      backgroundColor: '#222',
      borderRadius: 16,
      boxShadow: '0px 0px 20px -10px black',
      userSelect: 'none' // Prevents text selection while dragging
    }}>
      <div style={{
        textAlign: 'left',
        padding: '4px 8px',
        fontWeight: 'bold',
        fontFamily: 'monospace',
        fontSize: 'small',
        cursor: 'grab',
        backgroundColor: '#333',
      userSelect: 'none'
      }}
        onMouseDown={(e) => {
          setIsDragging(true);
          dragStartRef.current = { x: e.clientX, y: e.clientY };
        }}
      >
        {title}
      </div>
      {children}
    </div>
  );
}

function CameraSensorWindow() {
  const {listen, stopListening} = useFabricProvider();

  const [imageSrc, setImageSrc] = useState(undefined)

  useEffect(() => {
    const listenerId = listen("sensors/camera", (x) => {
      setImageSrc(`data:image/png;base64,${x["rgb"]}`)
    })

    return () => stopListening("sensors/camera", listenerId);
  }, [listen, stopListening]);

  return <RobovizWindow title='sensors/camera'>
      <div style={{
        position: 'relative'
      }}>
        <img src={imageSrc} style={{
          width: '100%',
          height: '100%',
          objectFit: 'contain',
          userSelect: 'none'
        }} />
      </div>
  </RobovizWindow>
}

function VisionPipelineWindow() {
  const {listen, stopListening} = useFabricProvider();


  const [imageSrc, setImageSrc] = useState(undefined)
  const [imageSize, setImageSize] = useState(undefined)
  const [boundingBoxes, setBoundingBoxes] = useState<number[][]>([])

  useEffect(() => {
    const cameraListenerId = listen("sensors/camera", (x) => {
      setImageSrc(`data:image/png;base64,${x["rgb"]}`)
      setImageSize(x['shape'])
    })
    const visionBoundingBoxesId = listen("vision/bounding_boxes", (x) => {
      const binaryData = Uint8Array.fromBase64(x.predictions);
      const predictions = new Float32Array(binaryData.buffer);
      const flatArray = new Float32Array(predictions);

      const cols = 6;
      const rows = flatArray.length / cols;

      const nestedArray = [];
      for (let i = 0; i < rows; i++) {
          // Slice extracts 6 elements per row
          nestedArray.push(flatArray.slice(i * cols, (i + 1) * cols));
      }
      setBoundingBoxes(nestedArray)
    })

    return () => {
      stopListening("sensors/camera", cameraListenerId)
      stopListening("vision/bounding_boxes", visionBoundingBoxesId)
    };
  }, [listen, stopListening]);

  const scale = 500 / JSON.parse(imageSize ?? '[]')[1]

  return <RobovizWindow title='vision/bounding_box'>
    <div style={{
      position: 'relative'
    }}>
      <img src={imageSrc} style={{
        width: '100%',
        height: '100%',
        objectFit: 'contain',
        userSelect: 'none'
      }} />
      {boundingBoxes.filter(x => x[5] == 0).map((x, i) => <div key={`${i}`} style={{
        position: 'absolute',
        top: x[1] * scale,
        left: x[0] * scale,
        width: (x[2] - x[0]) * scale,
        height: (x[3] - x[1]) * scale,
        border: '3px solid blue'
      }} />)}
    </div>
    </RobovizWindow>
}

function App() {
  return (
    <>
      <CameraSensorWindow />
      <VisionPipelineWindow />
      <VisionPipelineWindow />
    </>
  )
}

export default App
