const uploadInput = document.getElementById("uploadInput");
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

// Handle image upload
uploadInput.addEventListener("change", async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    // Reset state
    sensitiveBoxes = [];
    boxColors = [];
    blurredRegions = [];
    blurredMap.clear();
    
    // UI state
    uploadInput.disabled = true;
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
            const errorData = await res.json();
            throw new Error(errorData.detail || `Server error: ${res.status}`);
        }

        const data = await res.json();

        // Load original image
        img.onload = () => {
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

            drawImageWithBoxes();
            downloadBtn.style.display = "inline-block";
            loading.style.display = "none";
            uploadInput.disabled = false;
        };

        img.onerror = () => {
            throw new Error("Failed to load image");
        };

        img.src = `data:image/png;base64,${data.original_image}`;
        sensitiveBoxes = data.sensitive_boxes || [];
        boxColors = data.box_colors || [];

        // Load backend blurred regions as Image objects
        blurredRegions = [];
        (data.blurred_regions || []).forEach((b64, i) => {
            const regionImg = new Image();
            regionImg.onload = () => {
                blurredRegions[i] = regionImg;
                // Redraw when new region loads
                if (i === data.blurred_regions.length - 1) {
                    drawImageWithBoxes();
                }
            };
            regionImg.src = `data:image/png;base64,${b64}`;
        });

    } catch (err) {
        console.error("Error uploading image:", err);
        loading.style.display = "none";
        uploadInput.disabled = false;
        
        if (errorDiv) {
            errorDiv.textContent = err.message || "Failed to process image. Please try again.";
            errorDiv.style.display = "block";
        }
    }
});

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