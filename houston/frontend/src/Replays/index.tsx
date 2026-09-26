import {
    listReplays,
    type Replay,
    type ReplayListResponse,
} from "../api/api.ts";
import { useEffect, useState } from "react";
import { THEME } from "../Roboviz/utils.tsx";
import { UploadForm } from "./UploadForm.tsx";
import { ReplayCard } from "./ReplayCard.tsx";
import { ReplayDetail } from "./ReplayDetail.tsx";

export function Replays() {
    const [replays, setReplays] = useState<ReplayListResponse>();
    const [selectedReplays, setSelectedReplays] = useState<Replay[]>([]);

    useEffect(() => {
        listReplays({ page: 0 }).then((x) => {
            setReplays(x);
        });
    }, []);

    return (
        <div
            style={{
                display: "grid",
                gridTemplateColumns: "1fr 400px",
                height: "100vh",
                background: "#0d0d0d",
                color: "#eee",
            }}
        >
            <div
                style={{
                    display: "flex",
                    flexDirection: "row",
                    flexWrap: "wrap",
                    gap: "16px",
                    padding: 24,
                    alignContent: "flex-start",
                    overflowY: "auto",
                }}
            >
                {replays?.replays.map((replay) => (
                    <ReplayCard
                        key={replay.replay_id}
                        replay={replay}
                        selected={selectedReplays
                            .map((x) => x.replay_id)
                            .includes(replay.replay_id)}
                        onClick={(e) => {
                            let selectedItems = [...selectedReplays];
                            const isSelected = selectedItems.some(
                                (x) => x.replay_id === replay.replay_id,
                            );

                            if (e.metaKey || e.ctrlKey) {
                                if (isSelected) {
                                    selectedItems = selectedItems.filter(
                                        (x) => x.replay_id !== replay.replay_id,
                                    );
                                } else {
                                    selectedItems.push(replay);
                                }
                            } else {
                                selectedItems = [replay];
                            }

                            setSelectedReplays(selectedItems);
                        }}
                    />
                ))}

                <div
                    style={{
                        width: "100%",
                        padding: "20px 0",
                        textAlign: "center",
                        color: "#666",
                        fontSize: "14px",
                        borderTop: "1px solid #333",
                        marginTop: 16,
                    }}
                >
                    Total Replays: {replays?.total ?? 0}
                </div>
            </div>

            <div
                style={{
                    padding: 24,
                    display: "flex",
                    flexDirection: "column",
                    gap: "24px",
                    borderLeft: `1px solid #333`,
                    background: "#121212",
                    overflowY: "auto",
                }}
            >
                <div
                    style={{
                        fontSize: "20px",
                        fontWeight: "bold",
                        color: THEME,
                        borderBottom: `2px solid ${THEME}`,
                        paddingBottom: 8,
                        marginBottom: 8,
                    }}
                >
                    {selectedReplays.length === 0
                        ? "Upload Replay"
                        : selectedReplays.length === 1
                          ? "Replay Details"
                          : `${selectedReplays.length} Replays Selected`}
                </div>
                <div>
                    {selectedReplays.length === 0 ? (
                        <UploadForm />
                    ) : (
                        <ReplayDetail
                            replays={selectedReplays}
                            onTagsUpdated={(updatedList) => {
                                setSelectedReplays(updatedList);
                                listReplays({ page: 0 }).then((x) => {
                                    setReplays(x);
                                });
                            }}
                            onDelete={() => {
                                setSelectedReplays([]);
                                listReplays({ page: 0 }).then((x) => {
                                    setReplays(x);
                                });
                            }}
                        />
                    )}
                </div>
            </div>
        </div>
    );
}
