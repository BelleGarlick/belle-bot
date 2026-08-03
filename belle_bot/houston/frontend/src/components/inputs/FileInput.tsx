import { Button } from "./Button.tsx";

export function FileInput({
    label,
    files,
    setFiles,
    multi = false,
}: {
    label?: string;
    files?: File[];
    setFiles: (files: File[] | undefined) => void;
    multi?: boolean;
}) {
    return (
        <div
            style={{
                display: "flex",
                flexDirection: "column",
                gap: 4,
            }}
        >
            {label && <span>{label}</span>}
            <Button
                style={{
                    position: "relative",
                    overflow: "hidden",
                }}
            >
                Select File...
                <input
                    type="file"
                    style={{
                        position: "absolute",
                        top: 0,
                        left: 0,
                        width: "100%",
                        height: "100%",
                        scale: 100,
                        opacity: 0,
                    }}
                    multiple={multi}
                    onChange={(e) => {
                        const items: File[] = [];
                        for (
                            let i = 0;
                            i < (e.currentTarget.files?.length ?? 0);
                            i++
                        ) {
                            items.push(e.currentTarget.files!.item(i)!);
                        }

                        setFiles(items);
                    }}
                />
            </Button>
        </div>
    );
}
