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
    return (
        <div
            onClick={onClick}
            style={{
                display: "block",
                textDecoration: "none",
                color: "inherit",
                width: 400,
                background: "black",
                borderRadius: 8,
                padding: 16,
                boxSizing: "border-box",
                outlineWidth: selected ? 2 : 0,
                outlineStyle: "solid",
                outlineColor: THEME,
                fontSize: 14,
            }}
        >
            <b style={{ color: THEME }}>{replay.replay_id}</b>
            <Tags tags={replay.tags ?? []} />
        </div>
    );
}
