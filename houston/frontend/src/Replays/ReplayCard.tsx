import type { Replay } from "../api/api.ts";
import { THEME } from "../Roboviz/utils.tsx";
import type { MouseEventHandler } from "react";
import { Tags } from "../components/tags/Tags.tsx";

export function ReplayCard({
    replay,
    selected,
    onClick,
}: {
    replay: Replay;
    selected: boolean;
    onClick: (event: MouseEventHandler<HTMLDivElement>) => void;
}) {
    const uploadDate = new Date(replay.upload_time).toLocaleDateString();

    return (
        <div
            onClick={onClick}
            style={{
                display: "flex",
                flexDirection: "column",
                gap: 8,
                textDecoration: "none",
                color: "inherit",
                width: 320,
                background: "#1a1a1a",
                borderRadius: 12,
                padding: 16,
                boxSizing: "border-box",
                border: selected ? `2px solid ${THEME}` : "2px solid transparent",
                boxShadow: selected
                    ? `0 0 15px ${THEME}40`
                    : "0 4px 6px rgba(0,0,0,0.3)",
                transition: "all 0.2s ease-in-out",
                cursor: "pointer",
                fontSize: 14,
            }}
            onMouseOver={(e) => {
                if (!selected)
                    e.currentTarget.style.borderColor = `${THEME}80`;
            }}
            onMouseOut={(e) => {
                if (!selected) e.currentTarget.style.borderColor = "transparent";
            }}
        >
            <div
                style={{
                    display: "flex",
                    justifyContent: "space-between",
                    alignItems: "flex-start",
                }}
            >
                <b
                    style={{
                        color: THEME,
                        wordBreak: "break-all",
                        fontSize: "12px",
                        fontFamily: "monospace",
                    }}
                >
                    {replay.replay_id.split("-")[0]}...
                </b>
                <span style={{ color: "#888", fontSize: "12px" }}>
                    {uploadDate}
                </span>
            </div>
            {replay.filename && (
                <div
                    style={{
                        fontWeight: "bold",
                        whiteSpace: "nowrap",
                        overflow: "hidden",
                        textOverflow: "ellipsis",
                    }}
                >
                    {replay.filename}
                </div>
            )}
            <div style={{ color: "#aaa", fontSize: "12px" }}>
                {replay.platform || "Unknown Platform"}
            </div>
            <Tags tags={replay.tags ?? []} />
        </div>
    );
}
