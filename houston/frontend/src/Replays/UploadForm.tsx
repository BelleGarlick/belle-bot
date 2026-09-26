import { useState } from "react";
import { Button, FileInput, TextField } from "../components/inputs";
import { uploadReplay } from "../api/api.ts";
import { TagsInput } from "../components/tags/TagsInput.tsx";
import {THEME} from "../Roboviz/utils.tsx";

export function UploadForm() {
    const [name, setName] = useState("");
    const [description, setDescription] = useState("");
    const [platform, setPlatform] = useState("belle-bot v0");
    const [permanent, setPermanent] = useState(false);
    const [tags, setTags] = useState<string[]>([]);
    const [files, setFiles] = useState<File[]>();
    const [uploadingStatus, setUploadingStatus] = useState<string | null>(null);
    const [uploadResult, setUploadResult] = useState<{
        success: boolean;
        message: string;
    } | null>(null);

    const onUpload = async () => {
        if (!files || files.length === 0) {
            setUploadResult({ success: false, message: "Please select at least one file" });
            return;
        }

        setUploadResult(null);
        setUploadingStatus(`Uploading ${files.length} file(s)...`);
        
        const uploadPromises = files.map(async (file) => {
            try {
                const res = await uploadReplay({
                    file: file,
                    filename: name.length > 0 ? name : null,
                    description: description,
                    platform,
                    permanent: permanent,
                    tags: tags,
                });
                return { file, res };
            } catch (error) {
                return { file, error };
            }
        });

        const results = await Promise.all(uploadPromises);
        
        let successCount = 0;
        let failCount = 0;
        const errors: string[] = [];

        results.forEach((result) => {
            if ('res' in result && result.res.status === 200) {
                successCount++;
            } else {
                failCount++;
                const errorMessage = 'res' in result 
                    ? JSON.stringify(result.res.data)
                    : (result.error as any)?.message || "Unknown error";
                errors.push(`${result.file.name}: ${errorMessage}`);
            }
        });

        setUploadingStatus(null);
        if (failCount === 0) {
            setUploadResult({
                success: true,
                message: `Successfully uploaded ${successCount} file(s)!`,
            });
            setFiles(undefined);
            setName("");
            setDescription("");
            setTags([]);
        } else {
            setUploadResult({
                success: false,
                message: `Uploaded ${successCount} files. Failed ${failCount} files:\n` + errors.join("\n"),
            });
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
            <FileInput label="Select Files" files={files} setFiles={setFiles} multi />
            {files && files.length > 0 && (
                <div
                    style={{
                        fontSize: "13px",
                        color: "#ccc",
                        background: "#0d0d0d",
                        padding: "10px 14px",
                        borderRadius: 6,
                        maxHeight: "120px",
                        overflowY: "auto",
                        border: "1px solid #333",
                    }}
                >
                    <div style={{ marginBottom: 6, fontWeight: "bold", color: "#888", fontSize: "11px", textTransform: "uppercase" }}>
                        Selected Files ({files.length})
                    </div>
                    {files.map((f, i) => (
                        <div key={i} style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
                            <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{f.name}</span>
                            <span style={{ color: "#666", marginLeft: 8, fontSize: "11px" }}>
                                {(f.size / 1024).toFixed(1)} KB
                            </span>
                        </div>
                    ))}
                </div>
            )}
            <div
                style={{
                    display: "grid",
                    gridTemplateColumns: "1fr 1fr",
                    gap: 12,
                }}
            >
                <div style={{ gridColumn: "span 2" }}>
                    <TextField label="Name" value={name} onChange={setName} />
                </div>
                <div style={{ gridColumn: "span 2" }}>
                    <TextField
                        label="Description"
                        value={description}
                        onChange={setDescription}
                        multiline
                    />
                </div>
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
                    style={{ 
                        width: 18, 
                        height: 18, 
                        cursor: "pointer",
                        accentColor: THEME
                    }}
                />
                <label
                    htmlFor="permanent-upload"
                    style={{ 
                        fontSize: 14, 
                        cursor: "pointer",
                        color: "#eee",
                        userSelect: "none"
                    }}
                >
                    Permanent Replay
                </label>
            </div>
            {uploadResult && (
                <div
                    style={{
                        padding: "12px",
                        borderRadius: 6,
                        fontSize: "14px",
                        backgroundColor: uploadResult.success ? "rgba(40, 167, 69, 0.15)" : "rgba(220, 53, 69, 0.15)",
                        color: uploadResult.success ? "#51cf66" : "#ff6b6b",
                        border: `1px solid ${uploadResult.success ? "#2f4535" : "#4b2e2e"}`,
                        whiteSpace: "pre-wrap",
                        wordBreak: "break-word"
                    }}
                >
                    {uploadResult.message}
                </div>
            )}
            <div
                style={{
                    marginTop: 4,
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
