import { useState } from "react";
import { THEME } from "../../Roboviz/utils.tsx";

export function TagsInput({
    label,
    tags,
    onChange,
}: {
    label: string;
    tags: string[];
    onChange: (tags: string[]) => void;
}) {
    const [inputValue, setInputValue] = useState("");

    const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
        if (e.key === "Enter") {
            e.preventDefault();
            const trimmed = inputValue.trim();
            if (trimmed && !tags.includes(trimmed)) {
                onChange([...tags, trimmed]);
                setInputValue("");
            }
        }
    };

    const removeTag = (indexToRemove: number) => {
        onChange(tags.filter((_, index) => index !== indexToRemove));
    };

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
                placeholder="Press Enter to add tags"
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                onKeyDown={handleKeyDown}
                style={{
                    border: `2px solid ${THEME}`,
                    borderRadius: 8,
                    padding: "10px 18px",
                    color: THEME,
                    outline: "none",
                    fontStyle: "monospace",
                }}
            />
            <div
                style={{
                    display: "flex",
                    flexWrap: "wrap",
                    gap: 8,
                    marginTop: 4,
                }}
            >
                {tags.map((tag, index) => (
                    <div
                        key={index}
                        style={{
                            display: "flex",
                            alignItems: "center",
                            gap: 6,
                            backgroundColor: `${THEME}15`,
                            border: `1px solid ${THEME}`,
                            borderRadius: 8,
                            padding: "2px 10px",
                            fontSize: "14px",
                            fontFamily: "monospace",
                            color: THEME,
                        }}
                    >
                        <span>{tag}</span>
                        <span
                            onClick={() => removeTag(index)}
                            style={{
                                cursor: "pointer",
                                fontWeight: "bold",
                                paddingLeft: 2,
                            }}
                        >
                            &times;
                        </span>
                    </div>
                ))}
            </div>
        </div>
    );
}
