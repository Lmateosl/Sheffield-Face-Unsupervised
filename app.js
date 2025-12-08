// Configuration
const API_BASE_URL = 'http://localhost:8000';

// DOM Elements
const uploadArea = document.getElementById('uploadArea');
const imageInput = document.getElementById('imageInput');
const previewCanvas = document.getElementById('previewCanvas');
const predictBtn = document.getElementById('predictBtn');
const resultsSection = document.getElementById('resultsSection');
const statusMessage = document.getElementById('statusMessage');
const loadingIndicator = document.getElementById('loadingIndicator');

// State
let uploadedFile = null;

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    setupEventListeners();
    checkAPIHealth();
});

function setupEventListeners() {
    // Click to upload
    uploadArea.addEventListener('click', () => {
        if (!uploadedFile) {
            imageInput.click();
        }
    });

    // File input change
    imageInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            handleFileUpload(file);
        }
    });

    // Drag and drop
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        const file = e.dataTransfer.files[0];
        if (file && file.type.startsWith('image/')) {
            handleFileUpload(file);
        } else {
            showStatus('Please upload an image file', 'error');
        }
    });

    // Predict button
    predictBtn.addEventListener('click', () => {
        if (uploadedFile) {
            predictImage();
        }
    });
}

function handleFileUpload(file) {
    uploadedFile = file;
    
    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        const img = new Image();
        img.onload = () => {
            displayImagePreview(img);
            predictBtn.disabled = false;
            showStatus('Image loaded. Click "Classify Image" to process.', 'info');
        };
        img.src = e.target.result;
    };
    reader.readAsDataURL(file);
}

function displayImagePreview(img) {
    const ctx = previewCanvas.getContext('2d');
    
    // Calculate dimensions to maintain aspect ratio
    const maxWidth = 400;
    const maxHeight = 400;
    let width = img.width;
    let height = img.height;
    
    if (width > height) {
        if (width > maxWidth) {
            height *= maxWidth / width;
            width = maxWidth;
        }
    } else {
        if (height > maxHeight) {
            width *= maxHeight / height;
            height = maxHeight;
        }
    }
    
    previewCanvas.width = width;
    previewCanvas.height = height;
    ctx.drawImage(img, 0, 0, width, height);
    
    // Show canvas, hide placeholder
    document.querySelector('.upload-placeholder').hidden = true;
    previewCanvas.hidden = false;
}

async function predictImage() {
    if (!uploadedFile) {
        showStatus('Please upload an image first', 'error');
        return;
    }

    // Show loading
    loadingIndicator.hidden = false;
    predictBtn.disabled = true;
    resultsSection.hidden = true;
    hideStatus();

    try {
        // Create FormData
        const formData = new FormData();
        formData.append('file', uploadedFile);

        // Send request
        const response = await fetch(`${API_BASE_URL}/predict-image`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Prediction failed');
        }

        const result = await response.json();
        displayResults(result);
        showStatus('Classification completed successfully!', 'success');

    } catch (error) {
        console.error('Prediction error:', error);
        showStatus(`Error: ${error.message}. Make sure the API server is running.`, 'error');
    } finally {
        loadingIndicator.hidden = true;
        predictBtn.disabled = false;
    }
}

function displayResults(result) {
    // Show results section
    resultsSection.hidden = false;

    // Display main results
    document.getElementById('predictedLabel').textContent = result.predicted_label_0_based;
    document.getElementById('predictedGroup').textContent = result.predicted_group_1_based;
    
    // Calculate confidence (max probability)
    const maxProb = Math.max(...result.probabilities);
    document.getElementById('confidence').textContent = `${(maxProb * 100).toFixed(2)}%`;

    // Display probability grid
    displayProbabilities(result.probabilities, result.predicted_label_0_based);

    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function displayProbabilities(probabilities, predictedLabel) {
    const grid = document.getElementById('probabilityGrid');
    grid.innerHTML = '';

    probabilities.forEach((prob, index) => {
        const item = document.createElement('div');
        item.className = 'probability-item';
        
        // Highlight the predicted class
        if (index === predictedLabel) {
            item.classList.add('top-prediction');
        }

        item.innerHTML = `
            <div class="probability-label">Person ${index}</div>
            <div class="probability-value">${(prob * 100).toFixed(1)}%</div>
            <div class="probability-bar">
                <div class="probability-fill" style="width: ${prob * 100}%"></div>
            </div>
        `;

        grid.appendChild(item);
    });
}

async function checkAPIHealth() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        if (response.ok) {
            console.log('API is healthy');
        } else {
            showStatus('Warning: API server may not be running properly', 'error');
        }
    } catch (error) {
        showStatus('Warning: Cannot connect to API server. Please start the backend server.', 'error');
    }
}

function showStatus(message, type) {
    statusMessage.textContent = message;
    statusMessage.className = `status-message ${type} show`;
}

function hideStatus() {
    statusMessage.className = 'status-message';
}

// Reset functionality
function resetUpload() {
    uploadedFile = null;
    previewCanvas.hidden = true;
    document.querySelector('.upload-placeholder').hidden = false;
    imageInput.value = '';
    predictBtn.disabled = true;
    resultsSection.hidden = true;
    hideStatus();
}

// Make reset available globally if needed
window.resetUpload = resetUpload;
