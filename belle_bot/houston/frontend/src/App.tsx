import "./App.css";
import { RoboViz } from "./Roboviz";
import { NavBar } from "./Navbar";
import { Replays } from "./Replays";
import { ReplayDetail } from "./Replays/ReplayDetail.tsx";
import { Routes, Route, Navigate } from "react-router-dom";

function App() {
    return (
        <div style={{ height: "100vh" }}>
            <NavBar />

            <div style={{ height: "calc(100% - 50px)" }}>
                <Routes>
                    <Route
                        path="/roboviz"
                        element={<RoboViz style={{ height: "100%" }} />}
                    />
                    <Route path="/replays" element={<Replays />} />
                    <Route
                        path="/replays/:replayId"
                        element={<ReplayDetail />}
                    />
                    <Route
                        path="*"
                        element={<Navigate to="/replays" replace />}
                    />
                </Routes>
            </div>
        </div>
    );
}

export default App;
