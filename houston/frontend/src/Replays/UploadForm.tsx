import { useState } from "react";
import { Button, FileInput, TextField } from "../components/inputs";
import { uploadReplay } from "../api/api.ts";
import { TagsInput } from "../components/tags/TagsInput.tsx";

export function UploadForm() {
    const [name, setName] = useState("");
    const [description, setDescription] = useState("");
    const [platform, setPlatform] = useState("belle-bot v0");
    const [permanent, setPermanent] = useState(false);
    const [tags, setTags] = useState<string[]>([]);
    const [files, setFiles] = useState<File[]>();
    const [uploadingStatus, setUploadingStatus] = useState<string | null>(null);

    const onUpload = async () => {
        if (!files || files.length === 0) {
            alert("Please select a file");
            return;
        }

        if (files.length === 1) {
            setUploadingStatus("Uploading 1/1...");
            const file = files[0];
            const res = await uploadReplay({
                file: file,
                filename: name.length > 0 ? name : null,
                description: description,
                platform,
                permanent: permanent,
                tags: tags,
            });

            setUploadingStatus(null);
            if (res.status === 200) {
                alert("Upload successful");
            } else {
                alert("Upload failed: " + JSON.stringify(res.data));
            }
            return;
        }

        let successCount = 0;
        let failCount = 0;
        const errors: string[] = [];

        for (let i = 0; i < files.length; i++) {
            const file = files[i];
            setUploadingStatus(`Uploading ${i + 1}/${files.length}...`);
            const res = await uploadReplay({
                file: file,
                filename: name.length > 0 ? name : null,
                description: description,
                platform,
                permanent: permanent,
                tags: tags,
            });

            if (res.status === 200) {
                successCount++;
            } else {
                failCount++;
                errors.push(`${file.name}: ${JSON.stringify(res.data)}`);
            }
        }

        setUploadingStatus(null);
        if (failCount === 0) {
            alert("Upload successful");
        } else {
            alert(
                `Uploaded ${successCount} files. Failed ${failCount} files:\n` +
                    errors.join("\n"),
            );
        }
    };

    return (
        <div
            style={{
                display: "flex",
                flexDirection: "column",
                gap: 16,
                background: "#1a1a1a",
                padding: 20,
                borderRadius: 12,
                border: "1px solid #333",
            }}
        >
            <FileInput files={files} setFiles={setFiles} multi />
            {files && files.length > 0 && (
                <div
                    style={{
                        fontSize: "12px",
                        color: "#888",
                        background: "#0d0d0d",
                        padding: "8px 12px",
                        borderRadius: 6,
                        maxHeight: "60px",
                        overflowY: "auto",
                    }}
                >
                    <b>Selected:</b> {files.map((f) => f.name).join(", ")}
                </div>
            )}
            <div
                style={{
                    display: "grid",
                    gridTemplateColumns: "1fr",
                    gap: 12,
                }}
            >
                <TextField label="Name" value={name} onChange={setName} />
                <TextField
                    label="Description"
                    value={description}
                    onChange={setDescription}
                    multiline
                />
                <TextField
                    label="Platform"
                    value={platform}
                    onChange={setPlatform}
                />
                <TagsInput label="Tags" tags={tags} onChange={setTags} />
            </div>
            <div
                style={{
                    display: "flex",
                    alignItems: "center",
                    gap: 10,
                    padding: "8px 0",
                }}
            >
                <input
                    type="checkbox"
                    id="permanent-upload"
                    checked={permanent}
                    onChange={(e) => setPermanent(e.target.checked)}
                    style={{ width: 18, height: 18, cursor: "pointer" }}
                />
                <label
                    htmlFor="permanent-upload"
                    style={{ fontSize: 14, cursor: "pointer" }}
                >
                    Permanent Replay
                </label>
            </div>
            <div
                style={{
                    marginTop: 8,
                    paddingTop: 16,
                    borderTop: "1px solid #333",
                }}
            >
                {uploadingStatus && (
                    <div
                        style={{
                            color: THEME,
                            fontWeight: "bold",
                            marginBottom: 12,
                            textAlign: "center",
                            fontSize: 14,
                        }}
                    >
                        {uploadingStatus}
                    </div>
                )}
                <Button
                    onClick={onUpload}
                    disabled={!!uploadingStatus}
                    style={{
                        width: "100%",
                        padding: "12px",
                        fontSize: "16px",
                        background: uploadingStatus ? "#333" : THEME,
                    }}
                >
                    {uploadingStatus ? "Uploading..." : "Upload Replay"}
                </Button>
            </div>
        </div>
    );
}
