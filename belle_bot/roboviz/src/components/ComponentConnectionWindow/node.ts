export interface DeltaTime {
    time: number
    delta: number
}

class Datum {
    public value: number = 0
}

export class Node {
    readonly name: string
    private spawnRateValue: number | undefined = undefined
    public readonly dependencies: Node[] = []

    public datums: Datum[] = []

    private lastSpawnTick = 0

    constructor(name: string) {
        this.name = name
    }

    require(node: Node) {
        node.dependencies.push(this)

        return this
    }

    spawnRate(rate: number) {
        this.spawnRateValue = rate

        return this
    }

    tick(t: DeltaTime) {
        if (this.spawnRateValue) {
            if (t.time - this.lastSpawnTick > this.spawnRateValue * 1000) {
                this.datums.push(new Datum())
                this.lastSpawnTick = t.time
            }
        }

        this.datums = this.datums
            .map((datum) => {
                datum.value += t.delta * 0.0005
                datum.value = Math.min(1, datum.value)
                return datum
            })
            .filter((x) => x.value <= 1)
    }
}
