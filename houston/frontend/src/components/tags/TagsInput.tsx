import { useState } from "react";
import { THEME } from "../../Roboviz/utils.tsx";
import { Tags } from "./Tags.tsx";

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
                gap: 6,
            }}
        >
            {label && (
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
            )}
            <input
                type="text"
                placeholder="Press Enter to add tags"
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                onKeyDown={handleKeyDown}
                style={{
                    border: `1px solid #444`,
                    borderRadius: 6,
                    padding: "10px 14px",
                    color: "#eee",
                    backgroundColor: "#0d0d0d",
                    outline: "none",
                    fontFamily: "inherit",
                    fontSize: "14px",
                    transition: "border-color 0.2s",
                }}
                onFocus={(e) => (e.currentTarget.style.borderColor = THEME)}
                onBlur={(e) => (e.currentTarget.style.borderColor = "#444")}
            />
            <Tags tags={tags} onRemove={removeTag} />
        </div>
    );
}
