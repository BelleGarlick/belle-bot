import "./App.css";
import { RoboViz } from "./Roboviz";
import { NavBar, Page } from "./Navbar";
import { useState } from "react";
import {Replays} from "./Replays";

function App() {
    const [page, setPage] = useState(Page.REPLAYS);

    return (
        <div style={{ height: "100vh" }}>
            <NavBar page={page} setPage={setPage} />

            <Replays />

            <RoboViz
                style={{
                    display: page === Page.ROBOVIZ ? "grid" : "none",
                    height: "100%",
                }}
            />
        </div>
    );
}

export default App;
