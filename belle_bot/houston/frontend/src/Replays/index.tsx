import { listReplaysReplaysGet } from "../api/api.ts";
import { useEffect, useState } from "react";
import { THEME } from "../Roboviz/utils.tsx";
import { UploadForm } from "./UploadForm.tsx";

export function Replays() {
    const [replays, setReplays] = useState([]);

    useEffect(() => {
        listReplaysReplaysGet({ page: 0 }).then((x) => setReplays(x.data));
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
                    flexDirection: "column",
                    gap: "16px",
                }}
            >
                {JSON.stringify(replays)}
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
                <UploadForm />
            </div>
        </div>
    );
}
