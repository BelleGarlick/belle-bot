import { useEffect, useState } from "react";
import {
    getReplayers,
    terminateReplayer,
    type ReplayerResponse,
} from "../api/api.ts";
import { THEME } from "../Roboviz/utils.tsx";
import styled from "@emotion/styled";
import { useFabricProvider } from "../Roboviz/contexts/ServerContext.tsx";
import { useNavigate } from "react-router-dom";

const Container = styled.div`
    padding: 20px;
    color: white;
`;

const ReplayerList = styled.div`
    display: flex;
    flex-direction: column;
    gap: 16px;
`;

const ReplayerItem = styled.div`
    border: 1px solid ${THEME};
    padding: 16px;
    border-radius: 8px;
    background-color: #1a1a1a;
    display: flex;
    justify-content: space-between;
    align-items: center;
`;

const Info = styled.div`
    display: flex;
    flex-direction: column;
    gap: 4px;
`;

const ButtonGroup = styled.div`
    display: flex;
    gap: 8px;
`;

const Button = styled.button<{ variant?: "danger" | "primary" }>`
    background-color: ${({ variant }) =>
        variant === "danger" ? "#ff4d4d" : THEME};
    color: white;
    border: none;
    padding: 8px 16px;
    border-radius: 4px;
    cursor: pointer;
    font-weight: bold;

    &:hover {
        background-color: ${({ variant }) =>
            variant === "danger" ? "#ff3333" : "#9922ee"};
    }
`;

export function Replayers() {
    const [replayers, setReplayers] = useState<ReplayerResponse[]>([]);
    const [loading, setLoading] = useState(true);
    const { setDomain } = useFabricProvider();
    const navigate = useNavigate();

    const fetchReplayers = () => {
        setLoading(true);
        getReplayers().then((x) => {
            setReplayers(x);
            setLoading(false);
        });
    };

    useEffect(() => {
        fetchReplayers();
        const interval = setInterval(fetchReplayers, 5000);
        return () => clearInterval(interval);
    }, []);

    const handleTerminate = (id: string) => {
        terminateReplayer(id).then(() => {
            fetchReplayers();
        });
    };

    const handleViewInRoboviz = (port: number) => {
        setDomain(`localhost:${port}`);
        navigate("/roboviz");
    };

    return (
        <Container>
            <h1>Active Replayers</h1>
            {loading && replayers.length === 0 ? (
                <div>Loading...</div>
            ) : replayers.length === 0 ? (
                <div>No active replayers.</div>
            ) : (
                <ReplayerList>
                    {replayers.map((replayer) => (
                        <ReplayerItem key={replayer.replayer_id}>
                            <Info>
                                <div>
                                    <strong>Name:</strong> {replayer.name}
                                </div>
                                <div>
                                    <strong>ID:</strong> {replayer.replayer_id}
                                </div>
                                <div>
                                    <strong>Port:</strong> {replayer.port}
                                </div>
                                <div>
                                    <strong>Replays:</strong>{" "}
                                    {replayer.replay_ids.join(", ")}
                                </div>
                                <div>
                                    <strong>Start Time:</strong>{" "}
                                    {new Date(
                                        replayer.start_time * 1000,
                                    ).toLocaleString()}
                                </div>
                            </Info>
                            <ButtonGroup>
                                <Button
                                    variant="primary"
                                    onClick={() =>
                                        handleViewInRoboviz(replayer.port)
                                    }
                                >
                                    View in RoboViz
                                </Button>
                                <Button
                                    variant="danger"
                                    onClick={() =>
                                        handleTerminate(replayer.replayer_id)
                                    }
                                >
                                    Terminate
                                </Button>
                            </ButtonGroup>
                        </ReplayerItem>
                    ))}
                </ReplayerList>
            )}
        </Container>
    );
}
