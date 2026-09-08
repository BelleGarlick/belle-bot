import { THEME } from "../../Roboviz/utils.tsx";

export function Tags({
    tags,
    onRemove,
}: {
    tags: string[];
    onRemove?: (idx: number) => void;
}) {
    return (
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
                        borderRadius: 12,
                        padding: "0px 10px",
                        fontSize: "12px",
                        fontFamily: "monospace",
                        color: THEME,
                    }}
                >
                    <span>{tag}</span>
                    {onRemove && (
                        <span
                            onClick={() => onRemove(index)}
                            style={{
                                cursor: "pointer",
                                fontWeight: "bold",
                                paddingLeft: 2,
                            }}
                        >
                            &times;
                        </span>
                    )}
                </div>
            ))}
        </div>
    );
}
