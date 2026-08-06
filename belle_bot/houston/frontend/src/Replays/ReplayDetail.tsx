import { type Replay, updateReplay } from "../api/api.ts";
import { type PropsWithChildren, useEffect, useState } from "react";
import { THEME } from "../Roboviz/utils.tsx";

function Labelled({ label, children }: { label: string } & PropsWithChildren) {
    return (
        <div
            style={{
                display: "flex",
                flexDirection: "column",
            }}
        >
            <span style={{ fontWeight: "bold" }}>{label}</span>
            {children}
        </div>
    );
}

function LabelledValue({
    label,
    value,
}: {
    label: string;
    value?: string | null;
}) {
    return (
        <div
            style={{
                display: "flex",
                flexDirection: "column",
            }}
        >
            <span style={{ fontWeight: "bold" }}>{label}</span>
            <span style={{ fontSize: 14 }}>{value}</span>
        </div>
    );
}

export function ReplayDetail({
    replays,
    onTagsUpdated,
}: {
    replays: Replay[];
    onTagsUpdated?: (updated: Replay[]) => void;
}) {
    // const { replayId } = useParams<{ replayId: string }>();
    // const [replay, setReplay] = useState<Replay | null>(null);
    // const [loading, setLoading] = useState(true);
    // const [error, setError] = useState<string | null>(null);
    //
    // useEffect(() => {
    //     if (!replayId) return;
    //     setLoading(true);
    //     getReplayInfoReplaysReplayIdInfoGet(replayId)
    //         .then((res) => {
    //             if (res.status === 200) {
    //                 setReplay(res.data);
    //             } else {
    //                 setError("Failed to fetch replay details");
    //             }
    //         })
    //         .catch((err) => {
    //             setError(err.message || "An error occurred");
    //         })
    //         .finally(() => {
    //             setLoading(false);
    //         });
    // }, [replayId]);
    //

    const [tags, setTags] = useState<string[]>([]);
    const [newTag, setNewTag] = useState("");

    useEffect(() => {
        const sharedTags = new Set(replays[0].tags ?? []);

        replays.forEach((r) => {
            sharedTags.forEach((t) => {
                if (!(r.tags ?? []).includes(t)) {
                    sharedTags.delete(t);
                }
            });
        });

        setTags([...sharedTags]);
    }, [replays]);

    const handleAddTag = async () => {
        const tagToAdd = newTag.trim();
        if (!tagToAdd) return;

        const updatedReplays = await Promise.all(
            replays.map(async (r) => {
                const rTags = r.tags ?? [];
                if (!rTags.includes(tagToAdd)) {
                    const newTags = [...rTags, tagToAdd];
                    const res = await updateReplay(r.replay_id, {
                        ...r,
                        tags: newTags,
                    });
                    if (res.status === 200) {
                        return res.data;
                    }
                }
                return r;
            }),
        );

        setNewTag("");
        if (onTagsUpdated) {
            onTagsUpdated(updatedReplays);
        }
    };

    const handleRemoveTag = async (tagToRemove: string) => {
        const updatedReplays = await Promise.all(
            replays.map(async (r) => {
                const rTags = r.tags ?? [];
                if (rTags.includes(tagToRemove)) {
                    const newTags = rTags.filter((t) => t !== tagToRemove);
                    const res = await updateReplay(r.replay_id, {
                        ...r,
                        tags: newTags,
                    });
                    if (res.status === 200) {
                        return res.data;
                    }
                }
                return r;
            }),
        );

        if (onTagsUpdated) {
            onTagsUpdated(updatedReplays);
        }
    };

    const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
        if (e.key === "Enter") {
            e.preventDefault();
            handleAddTag();
        }
    };

    return (
        <>
            <div
                style={{
                    display: "flex",
                    flexDirection: "column",
                    gap: "16px",
                }}
            >
                {replays.length === 1 && (
                    <>
                        <LabelledValue
                            label="Replay ID"
                            value={replays[0].replay_id}
                        />
                        <LabelledValue
                            label="Filename"
                            value={replays[0].filename}
                        />
                        <LabelledValue
                            label="Platform"
                            value={replays[0].platform}
                        />
                        <LabelledValue
                            label="Description"
                            value={replays[0].description}
                        />
                        <LabelledValue
                            label="Upload Time"
                            value={new Date(
                                replays[0].upload_time,
                            ).toLocaleString()}
                        />
                        <LabelledValue
                            label="Permanent"
                            value={replays[0].permanent ? "Yes" : "No"}
                        />
                    </>
                )}
                <Labelled label="Tags">
                    <div
                        style={{
                            display: "flex",
                            flexDirection: "column",
                            gap: 8,
                            marginTop: 4,
                        }}
                    >
                        <div style={{ display: "flex", gap: 8 }}>
                            <input
                                type="text"
                                value={newTag}
                                onChange={(e) => setNewTag(e.target.value)}
                                onKeyDown={handleKeyDown}
                                placeholder="Add a tag..."
                                style={{
                                    border: `2px solid ${THEME}`,
                                    borderRadius: 8,
                                    padding: "6px 12px",
                                    color: THEME,
                                    outline: "none",
                                    fontFamily: "monospace",
                                    fontSize: 14,
                                    flex: 1,
                                }}
                            />
                            <button
                                onClick={handleAddTag}
                                style={{
                                    cursor: "pointer",
                                    padding: "6px 16px",
                                    borderRadius: 8,
                                    border: `2px solid ${THEME}`,
                                    backgroundColor: THEME,
                                    color: "white",
                                    fontWeight: "bold",
                                    fontSize: 14,
                                }}
                            >
                                Add Tag
                            </button>
                        </div>
                        <div
                            style={{
                                display: "flex",
                                flexWrap: "wrap",
                                gap: 8,
                                marginTop: 4,
                            }}
                        >
                            {tags.length > 0 ? (
                                tags.map((tag) => (
                                    <div
                                        key={tag}
                                        style={{
                                            display: "flex",
                                            alignItems: "center",
                                            gap: 6,
                                            backgroundColor: `${THEME}15`,
                                            border: `1px solid ${THEME}`,
                                            borderRadius: 8,
                                            padding: "2px 10px",
                                            fontSize: "14px",
                                            fontFamily: "monospace",
                                            color: THEME,
                                        }}
                                    >
                                        <span>{tag}</span>
                                        <button
                                            onClick={() => handleRemoveTag(tag)}
                                            style={{
                                                border: "none",
                                                background: "none",
                                                cursor: "pointer",
                                                fontWeight: "bold",
                                                padding: "0 2px",
                                                fontSize: "14px",
                                                color: THEME,
                                            }}
                                        >
                                            &times;
                                        </button>
                                    </div>
                                ))
                            ) : (
                                <span style={{ fontSize: 14, color: "#666" }}>
                                    No tags
                                </span>
                            )}
                        </div>
                    </div>
                </Labelled>
            </div>
        </>
    );
}
