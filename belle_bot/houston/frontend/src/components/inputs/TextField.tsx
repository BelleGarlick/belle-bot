import {THEME} from "../../Roboviz/utils.tsx";

export function TextField({ label }: { label: string }) {
    return (
        <div
            style={{
                display: "flex",
                flexDirection: "column",
                gap: 4,
            }}
        >
            <span>{label}</span>
            <input
                type="text"
                style={{
                    border: `2px solid ${THEME}`,
                    borderRadius: 8,
                    padding: "10px 18px",
                    color: THEME,
                }}
            />
        </div>
    );
}
