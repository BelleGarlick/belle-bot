import { THEME } from "../../Roboviz/utils.tsx";

export function TextField({
    label,
    value,
    onChange,
    multiline = false,
}: {
    label: string;
    value?: string;
    onChange?: (value: string) => void;
    multiline?: boolean;
}) {
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
                value={value}
                onChange={(e) => onChange?.(e.target.value)}
                style={{
                    border: `2px solid ${THEME}`,
                    borderRadius: 8,
                    padding: "10px 18px",
                    color: THEME,
                    outline: "none",
                    fontStyle: "monospace",
                }}
            />
        </div>
    );
}
