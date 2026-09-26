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

import { TagsInput } from "../components/tags/TagsInput.tsx";

export function Replays() {
    const [replays, setReplays] = useState<ReplayListResponse>();
    const [selectedReplays, setSelectedReplays] = useState<Replay[]>([]);
    const [filterTags, setFilterTags] = useState<string[]>([]);

    useEffect(() => {
        listReplays({ page: 0 }).then((x) => {
            setReplays(x);
        });
    }, []);

    useEffect(() => {
        listReplays({ page: 0, tags: filterTags }).then((x) => {
            setReplays(x);
        });
    }, [filterTags]);

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
                    flexDirection: "column",
                    height: "100%",
                    overflow: "hidden",
                }}
            >
                <div
                    style={{
                        padding: "24px 24px 0 24px",
                        display: "flex",
                        alignItems: "center",
                        gap: 16,
                    }}
                >
                    <div style={{ flex: 1 }}>
                        <TagsInput
                            label=""
                            tags={filterTags}
                            onChange={setFilterTags}
                        />
                    </div>
                    <div
                        style={{
                            color: "#666",
                            fontSize: "14px",
                            fontFamily: "monospace",
                        }}
                    >
                        {replays?.total ?? 0} REPLAYS
                    </div>
                </div>
                <div
                    style={{
                        display: "flex",
                        flexDirection: "row",
                        flexWrap: "wrap",
                        gap: "16px",
                        padding: 24,
                        alignContent: "flex-start",
                        overflowY: "auto",
                        flex: 1,
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
                                            (x) =>
                                                x.replay_id !==
                                                replay.replay_id,
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
                        display: "flex",
                        justifyContent: "space-between",
                        alignItems: "center",
                        borderBottom: `2px solid ${THEME}`,
                        paddingBottom: 8,
                        marginBottom: 8,
                    }}
                >
                    <div
                        style={{
                            fontSize: "20px",
                            fontWeight: "bold",
                            color: THEME,
                        }}
                    >
                        {selectedReplays.length === 0
                            ? "Upload Replay"
                            : selectedReplays.length === 1
                              ? "Replay Details"
                              : `${selectedReplays.length} Replays Selected`}
                    </div>
                    {selectedReplays.length > 0 && (
                        <button
                            onClick={() => setSelectedReplays([])}
                            style={{
                                background: "none",
                                border: "none",
                                color: "#888",
                                cursor: "pointer",
                                fontSize: "12px",
                                textTransform: "uppercase",
                                letterSpacing: "0.5px",
                                padding: "4px 8px",
                                borderRadius: "4px",
                                transition: "all 0.2s",
                            }}
                            onMouseOver={(e) => {
                                e.currentTarget.style.color = "#eee";
                                e.currentTarget.style.background = "#333";
                            }}
                            onMouseOut={(e) => {
                                e.currentTarget.style.color = "#888";
                                e.currentTarget.style.background = "none";
                            }}
                        >
                            Clear
                        </button>
                    )}
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
