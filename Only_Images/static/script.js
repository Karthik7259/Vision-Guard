const uploadInput = document.getElementById("uploadInput");
const captureBtn = document.getElementById("captureBtn");
const canvas = document.getElementById("imageCanvas");
const ctx = canvas.getContext("2d");
const downloadBtn = document.getElementById("downloadBtn");
const loading = document.getElementById("loading");
const errorDiv = document.getElementById("error");

let sensitiveBoxes = [];
let boxColors = [];
let blurredRegions = [];
// Tracks which boxes are manually set to UNBLURRED (true means show original region)
let blurredMap = new Map();
let img = new Image();
let currentScale = 1;

// Improve UX: change cursor when hovering over a sensitive region
canvas.addEventListener("mousemove", (event) => {
    const rect = canvas.getBoundingClientRect();
    const x = (event.clientX - rect.left) * (canvas.width / rect.width);
    const y = (event.clientY - rect.top) * (canvas.height / rect.height);

    let over = false;
    for (let i = 0; i < sensitiveBoxes.length; i++) {
        const [x1, y1, w, h] = sensitiveBoxes[i];
        const scaledX1 = x1 * currentScale;
        const scaledY1 = y1 * currentScale;
        const scaledX2 = scaledX1 + (w * currentScale);
        const scaledY2 = scaledY1 + (h * currentScale);
        if (x >= scaledX1 && x <= scaledX2 && y >= scaledY1 && y <= scaledY2) {
            over = true;
            break;
        }
    }
    canvas.style.cursor = over ? "pointer" : "default";
});

// Shared: process a File/Blob by sending to backend and rendering result
async function processFile(file) {
    // Reset state
    sensitiveBoxes = [];
    boxColors = [];
    blurredRegions = [];
    blurredMap.clear();

    // UI state
    uploadInput.disabled = true;
    if (captureBtn) captureBtn.disabled = true;
    loading.style.display = "block";
    downloadBtn.style.display = "none";
    errorDiv.style.display = "none";

    const formData = new FormData();
    formData.append("file", file);

    try {
        const res = await fetch("http://127.0.0.1:8000/detect-pii/", {
            method: "POST",
            body: formData,
        });

        if (!res.ok) {
            const errorData = await res.json().catch(() => ({}));
            throw new Error(errorData.detail || `Server error: ${res.status}`);
        }

        const data = await res.json();

        // Load original image
        await new Promise((resolve, reject) => {
            img.onload = resolve;
            img.onerror = () => reject(new Error("Failed to load image"));
            img.src = `data:image/png;base64,${data.original_image}`;
        });

        // Scale canvas if image is too big
        const maxWidth = 800;
        const maxHeight = 600;
        let scale = 1;
        if (img.width > maxWidth || img.height > maxHeight) {
            scale = Math.min(maxWidth / img.width, maxHeight / img.height);
        }

        canvas.width = img.width * scale;
        canvas.height = img.height * scale;
        currentScale = scale;

        sensitiveBoxes = data.sensitive_boxes || [];
        boxColors = data.box_colors || [];

        // Load backend blurred regions as Image objects
        blurredRegions = [];
        const b64Regions = data.blurred_regions || [];
        await Promise.all(
            b64Regions.map((b64, i) => new Promise((resolve) => {
                const regionImg = new Image();
                regionImg.onload = () => { blurredRegions[i] = regionImg; resolve(); };
                regionImg.src = `data:image/png;base64,${b64}`;
            }))
        );

        drawImageWithBoxes();
        downloadBtn.style.display = "inline-block";
    } catch (err) {
        console.error("Error processing image:", err);
        if (errorDiv) {
            errorDiv.textContent = err.message || "Failed to process image. Please try again.";
            errorDiv.style.display = "block";
        }
    } finally {
        loading.style.display = "none";
        uploadInput.disabled = false;
        if (captureBtn) captureBtn.disabled = false;
    }
}

// Handle image upload
uploadInput.addEventListener("change", async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    await processFile(file);
});

// Handle screenshot capture via Screen Capture API
if (captureBtn) {
    captureBtn.addEventListener("click", async () => {
        try {
            // Some browsers allow without HTTPS for getDisplayMedia on localhost.
            const stream = await navigator.mediaDevices.getDisplayMedia({
                video: { cursor: "always" },
                audio: false,
            });

            const [track] = stream.getVideoTracks();
            const video = document.createElement("video");
            video.srcObject = stream;
            // Avoid showing it in the DOM; just wait for ready and grab a frame
            await video.play();
            // Small delay to ensure a frame is painted
            await new Promise(r => setTimeout(r, 150));

            const width = video.videoWidth;
            const height = video.videoHeight;
            const off = document.createElement("canvas");
            off.width = width;
            off.height = height;
            const octx = off.getContext("2d");
            octx.drawImage(video, 0, 0, width, height);

            // Stop sharing immediately after capture
            if (track) track.stop();
            if (video.srcObject) video.srcObject.getTracks().forEach(t => t.stop());

            off.toBlob(async (blob) => {
                if (!blob) throw new Error("Failed to capture screenshot");
                const file = new File([blob], "screenshot.png", { type: "image/png" });
                await processFile(file);
            }, "image/png");
        } catch (err) {
            // If user cancels share picker, just ignore; otherwise show error
            if (err && err.name === "NotAllowedError") {
                console.info("Screen capture cancelled by user");
                return;
            }
            console.error("Screen capture failed:", err);
            if (errorDiv) {
                errorDiv.textContent = err.message || "Screen capture failed.";
                errorDiv.style.display = "block";
            }
        }
    });
}

// Handle manual toggle on click
canvas.onclick = (event) => {
    const rect = canvas.getBoundingClientRect();
    const x = (event.clientX - rect.left) * (canvas.width / rect.width);
    const y = (event.clientY - rect.top) * (canvas.height / rect.height);

    for (let i = 0; i < sensitiveBoxes.length; i++) {
        const [x1, y1, w, h] = sensitiveBoxes[i];
        const scaledX1 = x1 * currentScale;
        const scaledY1 = y1 * currentScale;
        const scaledX2 = scaledX1 + (w * currentScale);
        const scaledY2 = scaledY1 + (h * currentScale);
        
        if (x >= scaledX1 && x <= scaledX2 && y >= scaledY1 && y <= scaledY2) {
            // Toggle: true means UNBLURRED (show original), false/undefined means blurred
            blurredMap.set(i, !blurredMap.get(i));
            drawImageWithBoxes();
            break;
        }
    }
};

// Draw image with backend and manual blur
function drawImageWithBoxes() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

    for (let i = 0; i < sensitiveBoxes.length; i++) {
        const [x1, y1, w, h] = sensitiveBoxes[i];
        const scaledX1 = x1 * currentScale;
        const scaledY1 = y1 * currentScale;
        const scaledW = w * currentScale;
        const scaledH = h * currentScale;

        const isUnblurred = !!blurredMap.get(i);

        // If NOT unblurred, draw the blurred region on top of the original
        if (!isUnblurred) {
            if (blurredRegions[i]) {
                ctx.drawImage(blurredRegions[i], scaledX1, scaledY1, scaledW, scaledH);
            } else {
                // Fallback: apply a canvas blur to the region
                ctx.save();
                ctx.beginPath();
                ctx.rect(scaledX1, scaledY1, scaledW, scaledH);
                ctx.clip();
                ctx.filter = "blur(8px)";
                ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
                ctx.restore();
            }

            // Single color for all blurred regions: yellow/gold overlay
            const overlay = "rgba(255,215,0,0.35)"; // gold/yellow
            ctx.fillStyle = overlay;
            ctx.fillRect(scaledX1, scaledY1, scaledW, scaledH);

            // Strong border for visibility (gold)
            ctx.strokeStyle = "#FFD700";
            ctx.lineWidth = 3;
            ctx.strokeRect(scaledX1, scaledY1, scaledW, scaledH);
        } else {
            // Unblurred: show original region with a subtle dashed outline (no fill)
            ctx.save();
            ctx.setLineDash([6, 4]);
            ctx.strokeStyle = "#00BFFF"; // deep sky blue for clarity
            ctx.lineWidth = 2;
            ctx.strokeRect(scaledX1, scaledY1, scaledW, scaledH);
            ctx.restore();
        }
    }
}

// Download final image
downloadBtn.addEventListener("click", () => {
    const link = document.createElement("a");
    link.download = "protected_image.png";
    link.href = canvas.toDataURL("image/png");
    link.click();
});

// Add some helpful instructions
console.log("VisionGuard Frontend Loaded");
console.log("• Upload an image to detect sensitive content");
console.log("• Click on red/yellow regions to toggle blur");
console.log("• Download the final protected image");