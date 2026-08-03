import { useState } from "react";
import { Button, FileInput, TextField, TagsInput } from "../components/inputs";
import { uploadReplayReplaysPost } from "../api/api.ts";

export function UploadForm() {
    const [name, setName] = useState("");
    const [description, setDescription] = useState("");
    const [platform, setPlatform] = useState("belle-bot v0");
    const [permanent, setPermanent] = useState(false);
    const [tags, setTags] = useState<string[]>([]);
    const [files, setFiles] = useState<File[]>();

    const onUpload = async () => {
        if (!files || files.length === 0) {
            alert("Please select a file");
            return;
        }

        if (files.length === 1) {
            const file = files[0];
            const res = await uploadReplayReplaysPost({
                file: file,
                filename: name.length > 0 ? name : null,
                description: description,
                platform,
                permanent: permanent,
                tags: tags,
            });

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

        for (const file of files) {
            const res = await uploadReplayReplaysPost({
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

        if (failCount === 0) {
            alert("Upload successful");
        } else {
            alert(`Uploaded ${successCount} files. Failed ${failCount} files:\n` + errors.join("\n"));
        }
    };

    return (
        <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
            <FileInput files={files} setFiles={setFiles} multi />
            {files && files.length > 0 && (
                <div style={{ fontSize: "14px", color: "#666" }}>
                    Selected files: {files.map((f) => f.name).join(", ")}
                </div>
            )}
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
            <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <input
                    type="checkbox"
                    checked={permanent}
                    onChange={(e) => setPermanent(e.target.checked)}
                />
                <span>Permanent</span>
            </div>
            <hr />
            <Button onClick={onUpload}>Upload</Button>
        </div>
    );
}
