import { type DeltaTime, Node } from './node.ts'

class SimulationData {
    static nodes = SimulationData.createNodes()

    public static createNodes() {
        const camera = new Node('sensor/camera').spawnRate(0.3)
        const imu = new Node('sensor/imu').spawnRate(0.15)
        const gps = new Node('sensor/gps').spawnRate(1)
        const segmentation = new Node('vision/segmentation').require(camera).spawnRate(1)
        const faces = new Node('vision/faces').require(camera).spawnRate(2)
        const position = new Node('mapping/position').require(gps).require(imu).spawnRate(0.15)
        const local = new Node('mapping/local')
            .require(camera)
            .require(position)
            .require(segmentation)
            .spawnRate(0.15)
        const pathplanning = new Node('planning/path').require(local).require(position).spawnRate(2)
        const llm = new Node('llm').require(faces).require(local).require(pathplanning).spawnRate(1)
        const movement = new Node('movement/drive')
            .require(pathplanning)
            .require(llm)
            .require(pathplanning)
            .spawnRate(0.1)
        const motors = new Node('hardware/motors').require(movement).spawnRate(0.1)

        const nodes = [
            camera,
            imu,
            gps,
            segmentation,
            faces,
            position,
            local,
            llm,
            motors,
            movement,
            pathplanning,
        ]

        return nodes
            .map((x) => ({ node: x, value: Math.random() }))
            .sort((a, b) => a.value - b.value)
            .map((x) => x.node)
    }
}

export function render(deltaTime: DeltaTime, windowSize, ctx: CanvasRenderingContext2D) {
    ctx.fillStyle = 'black'
    ctx.rect(0, 0, windowSize.width, windowSize.height)
    ctx.fill()

    const meshRadius = Math.min(windowSize.height, windowSize.width) * 0.3

    const getPositionOfNode = (node: Node) => {
        const idx = SimulationData.nodes.indexOf(node)

        return {
            x:
                meshRadius * Math.sin((idx / SimulationData.nodes.length) * 2 * Math.PI) +
                windowSize.width / 2,
            y:
                meshRadius * Math.cos((idx / SimulationData.nodes.length) * 2 * Math.PI) +
                windowSize.height / 2,
        }
    }

    const getTextPositionOfNode = (node: Node) => {
        const idx = SimulationData.nodes.indexOf(node)
        const d = meshRadius * 1.2

        return {
            x:
                d * Math.sin((idx / SimulationData.nodes.length) * 2 * Math.PI) +
                windowSize.width / 2,
            y:
                d * Math.cos((idx / SimulationData.nodes.length) * 2 * Math.PI) +
                windowSize.height / 2,
        }
    }

    for (let i = 0; i < SimulationData.nodes.length; i++) {
        SimulationData.nodes[i].tick(deltaTime)

        const pos = getPositionOfNode(SimulationData.nodes[i])
        const textPos = getTextPositionOfNode(SimulationData.nodes[i])

        ctx.beginPath()
        ctx.fillStyle = 'purple'
        ctx.arc(pos.x, pos.y, 10, 0, 2 * Math.PI)
        ctx.fill()

        ctx.textAlign = 'center'
        ctx.textBaseline = 'middle'
        ctx.font = '24px monospace'
        ctx.fillText(SimulationData.nodes[i].name, textPos.x, textPos.y)

        SimulationData.nodes[i].dependencies.forEach((b) => {
            const start = getPositionOfNode(SimulationData.nodes[i])
            const end = getPositionOfNode(b)

            ctx.beginPath()
            ctx.strokeStyle = 'purple'
            ctx.lineWidth = 2
            ctx.moveTo(start.x, start.y)
            ctx.lineTo(end.x, end.y)
            ctx.stroke()

            SimulationData.nodes[i].datums.forEach((d) => {
                const p = {
                    x: start.x + (end.x - start.x) * d.value,
                    y: start.y + (end.y - start.y) * d.value,
                }

                ctx.beginPath()
                ctx.fillStyle = 'white'
                ctx.arc(p.x, p.y, 3, 0, 2 * Math.PI)
                ctx.fill()
            })
        })
    }
}
