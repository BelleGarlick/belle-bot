import {
    type Replay,
    updateReplay,
    createReplayer,
    deleteReplay,
} from "../api/api.ts";
import { type PropsWithChildren, useEffect, useState } from "react";
import { THEME } from "../Roboviz/utils.tsx";
import { v4 } from "uuid";
import { Button } from "../components/inputs";

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
                padding: "8px 12px",
                background: "#1a1a1a",
                borderRadius: 8,
                border: "1px solid #333",
            }}
        >
            <span
                style={{
                    fontWeight: "bold",
                    fontSize: 12,
                    color: "#888",
                    textTransform: "uppercase",
                    letterSpacing: "0.5px",
                }}
            >
                {label}
            </span>
            <span
                style={{
                    fontSize: 14,
                    color: "#eee",
                    marginTop: 4,
                    wordBreak: "break-all",
                    fontFamily: label.includes("ID") ? "monospace" : "inherit",
                }}
            >
                {value || "N/A"}
            </span>
        </div>
    );
}

export function ReplayDetail({
    replays,
    onTagsUpdated,
    onDelete,
}: {
    replays: Replay[];
    onTagsUpdated?: (updated: Replay[]) => void;
    onDelete?: () => void;
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

    const runReplays = () => {
        const name =
            replays.length === 1
                ? replays[0].filename || "replay"
                : `replayer-${v4().split("-")[0]}`;
        createReplayer({
            name,
            replay_ids: replays.map((replay) => replay.replay_id),
        }).then((response) => {
            console.log(response);
            window.location.href = "/replayers";
        });
    };

    const handleDelete = async () => {
        if (replays.length === 0) return;
        const confirmMessage =
            replays.length === 1
                ? "Are you sure you want to delete this replay?"
                : `Are you sure you want to delete these ${replays.length} replays?`;

        if (!confirm(confirmMessage)) return;

        try {
            await Promise.all(replays.map((r) => deleteReplay(r.replay_id)));
            if (onDelete) {
                onDelete();
            }
        } catch (error) {
            console.error("Failed to delete replay(s):", error);
            alert("Failed to delete replay(s)");
        }
    };

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
                    return res;
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
                    return res;
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
        <div
            style={{
                display: "flex",
                flexDirection: "column",
                gap: "20px",
            }}
        >
            {replays.length === 1 && (
                <div
                    style={{
                        display: "grid",
                        gridTemplateColumns: "1fr 1fr",
                        gap: 12,
                    }}
                >
                    <div style={{ gridColumn: "span 2" }}>
                        <LabelledValue
                            label="Replay ID"
                            value={replays[0].replay_id}
                        />
                    </div>
                    <div style={{ gridColumn: "span 2" }}>
                        <LabelledValue
                            label="Filename"
                            value={replays[0].filename}
                        />
                    </div>
                    <LabelledValue
                        label="Platform"
                        value={replays[0].platform}
                    />
                    <LabelledValue
                        label="Permanent"
                        value={replays[0].permanent ? "Yes" : "No"}
                    />
                    <div style={{ gridColumn: "span 2" }}>
                        <LabelledValue
                            label="Upload Time"
                            value={new Date(
                                replays[0].upload_time,
                            ).toLocaleString()}
                        />
                    </div>
                    <div style={{ gridColumn: "span 2" }}>
                        <LabelledValue
                            label="Description"
                            value={replays[0].description}
                        />
                    </div>
                </div>
            )}
            <Labelled label="Tags">
                <div
                    style={{
                        display: "flex",
                        flexDirection: "column",
                        gap: 12,
                        marginTop: 4,
                        padding: 16,
                        background: "#1a1a1a",
                        borderRadius: 8,
                        border: "1px solid #333",
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
                                border: `1px solid #444`,
                                borderRadius: 6,
                                padding: "10px 14px",
                                color: "#eee",
                                backgroundColor: "#0d0d0d",
                                outline: "none",
                                fontFamily: "inherit",
                                fontSize: 14,
                                flex: 1,
                                transition: "border-color 0.2s",
                            }}
                            onFocus={(e) =>
                                (e.currentTarget.style.borderColor = THEME)
                            }
                            onBlur={(e) =>
                                (e.currentTarget.style.borderColor = "#444")
                            }
                        />
                        <Button
                            onClick={handleAddTag}
                            style={{
                                padding: "8px 16px",
                                fontSize: "14px",
                            }}
                        >
                            Add
                        </Button>
                    </div>
                    <div
                        style={{
                            display: "flex",
                            flexWrap: "wrap",
                            gap: 8,
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
                                        backgroundColor: `${THEME}25`,
                                        border: `1px solid ${THEME}`,
                                        borderRadius: 6,
                                        padding: "4px 10px",
                                        fontSize: "13px",
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
                                            fontSize: "16px",
                                            color: THEME,
                                            display: "flex",
                                            alignItems: "center",
                                        }}
                                    >
                                        &times;
                                    </button>
                                </div>
                            ))
                        ) : (
                            <span style={{ fontSize: 13, color: "#666" }}>
                                No shared tags
                            </span>
                        )}
                    </div>
                </div>
            </Labelled>

            <div
                style={{
                    display: "flex",
                    flexDirection: "column",
                    gap: 12,
                    marginTop: 8,
                    paddingTop: 20,
                    borderTop: "1px solid #333",
                }}
            >
                <Button
                    onClick={runReplays}
                    style={{
                        padding: "12px",
                        fontSize: "16px",
                        fontWeight: "bold",
                    }}
                >
                    ▶ Run Replayer
                </Button>
                <Button
                    onClick={handleDelete}
                    style={{
                        backgroundColor: "transparent",
                        border: "1px solid #dc3545",
                        color: "#dc3545",
                        padding: "10px",
                    }}
                    onMouseOver={(e) => {
                        e.currentTarget.style.backgroundColor = "#dc3545";
                        e.currentTarget.style.color = "white";
                    }}
                    onMouseOut={(e) => {
                        e.currentTarget.style.backgroundColor = "transparent";
                        e.currentTarget.style.color = "#dc3545";
                    }}
                >
                    Delete {replays.length > 1 ? `${replays.length} Replays` : "Replay"}
                </Button>
            </div>
        </div>
    );
}
