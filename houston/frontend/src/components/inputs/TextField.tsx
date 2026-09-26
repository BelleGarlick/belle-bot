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
    const inputStyle: React.CSSProperties = {
        border: `1px solid #444`,
        borderRadius: 6,
        padding: "10px 14px",
        color: "#eee",
        backgroundColor: "#0d0d0d",
        outline: "none",
        fontFamily: "inherit",
        fontSize: "14px",
        transition: "border-color 0.2s",
    };

    return (
        <div
            style={{
                display: "flex",
                flexDirection: "column",
                gap: 6,
            }}
        >
            <span
                style={{
                    fontSize: "12px",
                    fontWeight: "bold",
                    color: "#888",
                    textTransform: "uppercase",
                    letterSpacing: "0.5px",
                }}
            >
                {label}
            </span>
            {multiline ? (
                <textarea
                    value={value}
                    onChange={(e) => onChange?.(e.target.value)}
                    rows={3}
                    style={{
                        ...inputStyle,
                        resize: "vertical",
                        minHeight: "80px",
                    }}
                />
            ) : (
                <input
                    type="text"
                    value={value}
                    onChange={(e) => onChange?.(e.target.value)}
                    style={inputStyle}
                    onFocus={(e) => (e.currentTarget.style.borderColor = THEME)}
                    onBlur={(e) => (e.currentTarget.style.borderColor = "#444")}
                />
            )}
        </div>
    );
}
