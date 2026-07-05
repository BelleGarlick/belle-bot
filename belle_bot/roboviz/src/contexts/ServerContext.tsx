import { createContext, useContext, useState, type PropsWithChildren } from 'react'
import { v4 } from 'uuid'

type StreamCallback = (data: { [key: string]: unknown }) => string

interface FabricContextI {
    domain: string | undefined
    setDomain: (value: string | undefined) => void
    listen: (stream: string, callback: StreamCallback) => string
    stopListening: (stream: string, listenerId: string) => void
}

const FabricContext = createContext<FabricContextI | null>(null)

export function FabricContextProvider({ children }: PropsWithChildren) {
    const [domain, setDomain] = useState<string>('localhost:59990')

    const webSockets: { [key: string]: WebSocket } = {}
    const callbacks: { [key: string]: { [id: string]: StreamCallback } } = {}

    const listen = (stream: string, callback: StreamCallback) => {
        if (!domain) throw 'No domain set.'

        if (!Object.hasOwn(webSockets, stream)) {
            const socket = new WebSocket('ws://' + domain + '/listen/' + stream)

            socket.onmessage = (event: MessageEvent) => {
                const data = JSON.parse(event.data).data
                Object.values(callbacks[stream]).forEach((x) => x(data))
            }

            socket.onerror = console.log
            socket.onopen = console.log
            socket.onclose = console.log

            // todo handle socket events
            // todo handle errors if exists
            webSockets[stream] = socket
        }

        const newStreamId = v4()
        if (!Object.hasOwn(callbacks, stream)) callbacks[stream] = {}
        callbacks[stream][newStreamId] = callback

        return newStreamId
    }

    const stopListening = (stream: string, listenerId: string) => {
        if (!Object.hasOwn(callbacks, stream)) return
        if (!Object.hasOwn(callbacks[stream], listenerId)) return

        delete callbacks[stream][listenerId]
    }

    // todo create timer to kepe the sockets alive

    return (
        <FabricContext.Provider
            value={{
                domain,
                setDomain,
                listen,
                stopListening,
            }}
        >
            {children}
        </FabricContext.Provider>
    )
}

export const useFabricProvider = () => useContext(FabricContext)!
