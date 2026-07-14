import { listReplaysReplaysGet } from "../api/api.ts";
import { useEffect, useState } from "react";

// todo create upload form

export function Replays() {
    const [replays, setReplays] = useState([]);

    useEffect(() => {
        listReplaysReplaysGet({ page: 0 }).then((x) => setReplays(x.data));
    }, []);

    return <div>
        {JSON.stringify(replays)}

        name
        <input />
        description
        <input />
        description
        <input />
        <button>Upload</button>
    </div>;
}
