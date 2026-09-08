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
            <Tags tags={tags} onRemove={removeTag} />
        </div>
    );
}
