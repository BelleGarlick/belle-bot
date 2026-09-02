import {
    createContext,
    useContext,
    useState,
    useRef,
    useEffect,
    type PropsWithChildren,
} from "react";
import { v4 } from "uuid";

type StreamCallback = (data: { [key: string]: unknown }) => void;

interface FabricContextI {
    domain: string | undefined;
    setDomain: (value: string | undefined) => void;
    listen: (stream: string, callback: StreamCallback) => string;
    stopListening: (stream: string, listenerId: string) => void;
}

const FabricContext = createContext<FabricContextI | null>(null);

export function FabricContextProvider({ children }: PropsWithChildren) {
    const [domain, setDomain] = useState<string | undefined>("localhost:15401");

    const webSockets = useRef<{ [key: string]: WebSocket }>({});
    const callbacks = useRef<{
        [key: string]: { [id: string]: StreamCallback };
    }>({});

    const listen = (stream: string, callback: StreamCallback) => {
        if (!domain) throw "No domain set.";

        if (!Object.hasOwn(webSockets.current, stream)) {
            const path = "ws://" + domain + "/listen/" + stream;
            const socket = new WebSocket(path);

            socket.onmessage = (event: MessageEvent) => {
                const data = JSON.parse(event.data);
                if (callbacks.current[stream]) {
                    Object.values(callbacks.current[stream]).forEach((x) =>
                        x(data),
                    );
                }
            };

            socket.onerror = (x) => {
                console.error(`WebSocket error on stream ${stream}:`, x);
            };
            socket.onopen = () => {
                console.log(`WebSocket opened for stream ${stream} on ${domain}`);
            };
            socket.onclose = (x) => {
                console.log(`WebSocket closed for stream ${stream}:`, x);
                delete webSockets.current[stream];
            };

            webSockets.current[stream] = socket;
        }

        const newStreamId = v4();
        if (!Object.hasOwn(callbacks.current, stream))
            callbacks.current[stream] = {};
        callbacks.current[stream][newStreamId] = callback;

        return newStreamId;
    };

    const stopListening = (stream: string, listenerId: string) => {
        if (!Object.hasOwn(callbacks.current, stream)) return;
        if (!Object.hasOwn(callbacks.current[stream], listenerId)) return;

        delete callbacks.current[stream][listenerId];

        if (Object.keys(callbacks.current[stream]).length === 0) {
            if (webSockets.current[stream]) {
                webSockets.current[stream].close();
                delete webSockets.current[stream];
            }
        }
    };

    useEffect(() => {
        // When domain changes, close all existing sockets so they can be re-opened on the new domain
        const sockets = webSockets.current;
        return () => {
            Object.values(sockets).forEach((s) => s.close());
            webSockets.current = {};
        };
    }, [domain]);

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
    );
}

export const useFabricProvider = () => useContext(FabricContext)!;
