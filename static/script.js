const fileInput = document.getElementById("image-input");
const fileNameLabel = document.getElementById("file-name");
const dropzone = document.getElementById("dropzone");

if (fileInput) {
    fileInput.addEventListener("change", (event) => {
        const file = event.target.files[0];
        if (!file) {
            fileNameLabel.textContent = "No file selected yet.";
            return;
        }
        fileNameLabel.textContent = `${file.name} · ${(file.size / 1024).toFixed(1)} KB`;
    });
}

if (dropzone) {
    dropzone.addEventListener("dragover", (event) => {
        event.preventDefault();
        dropzone.classList.add("hover");
    });

    dropzone.addEventListener("dragleave", () => dropzone.classList.remove("hover"));
    dropzone.addEventListener("drop", (event) => {
        event.preventDefault();
        dropzone.classList.remove("hover");
        const dt = event.dataTransfer;
        if (dt?.files?.length) {
            fileInput.files = dt.files;
            const file = dt.files[0];
            fileNameLabel.textContent = `${file.name} · ${(file.size / 1024).toFixed(1)} KB`;
        }
    });
}

// Lazy-load analysis report on demand
const showReportBtn = document.getElementById("show-report");
const reportCard = document.getElementById("analysis-report");
const reportFrame = document.getElementById("report-frame");

if (showReportBtn && reportCard && reportFrame) {
    showReportBtn.addEventListener("click", () => {
        if (reportCard.classList.contains("collapsed")) {
            const url = reportFrame.getAttribute("data-report-url");
            if (url) {
                reportFrame.setAttribute("src", url);
            }
            reportCard.classList.remove("collapsed");
            reportCard.scrollIntoView({ behavior: "smooth", block: "start" });
        } else {
            reportCard.classList.add("collapsed");
            reportFrame.setAttribute("src", "");
        }
    });
}
