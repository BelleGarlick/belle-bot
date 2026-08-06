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
            if (x.status === 200) {
                setReplays(x.data);
            }
        });
    }, []);

    return (
        <div
            style={{
                display: "grid",
                gridTemplateColumns: "auto 360px",
                height: "100%",
            }}
        >
            <div
                style={{
                    display: "flex",
                    flexDirection: "row",
                    flexWrap: "wrap",
                    gap: "8px",
                    padding: 16,
                    alignItems: "center",
                }}
            >
                {replays?.replays.map((replay) => (
                    <ReplayCard
                        replay={replay}
                        selected={selectedReplays
                            .map((x) => x.replay_id)
                            .includes(replay.replay_id)}
                        onClick={(e) => {
                            let selectedItems = [];

                            if (e.metaKey || e.ctrlKey) {
                                selectedItems = selectedReplays.filter(
                                    (x) => x.replay_id != replay.replay_id,
                                );
                            }

                            // if (e.shiftKey) {
                            // todo select items until this one

                            setSelectedReplays([...selectedItems, replay]);
                        }}
                    />
                ))}

                <div>{JSON.stringify(replays?.total ?? 0)}</div>
            </div>

            <div
                style={{
                    padding: 16,
                    display: "flex",
                    flexDirection: "column",
                    gap: "16px",
                    borderLeft: `2px solid ${THEME}`,
                }}
            >
                {selectedReplays.length === 0 ? (
                    <UploadForm />
                ) : (
                    <ReplayDetail
                        replays={selectedReplays}
                        onTagsUpdated={(updatedList) => {
                            setSelectedReplays(updatedList);
                            listReplaysReplaysGet({ page: 0 }).then((x) => {
                                if (x.status === 200) {
                                    setReplays(x.data);
                                }
                            });
                        }}
                    />
                )}
            </div>
        </div>
    );
}
