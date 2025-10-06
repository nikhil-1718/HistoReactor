// script.js - Updated histogram with two distinct colors (blue for original, green for enhanced)
let filenames = [];
let outputFilenames = [];
let originalHists = [];
let enhancedHists = [];
let currentIndex = 0;
let selectedMethod = '';
let histChart = null;

const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const previewSection = document.getElementById('previewSection');
const previewImage = document.getElementById('previewImage');
const prevPreview = document.getElementById('prevPreview');
const nextPreview = document.getElementById('nextPreview');
const previewCounter = document.getElementById('previewCounter');
const loader = document.getElementById('loader');
const resultsSection = document.getElementById('resultsSection');
const prevResult = document.getElementById('prevResult');
const nextResult = document.getElementById('nextResult');
const resultCounter = document.getElementById('resultCounter');
const comparisonSlider = document.getElementById('comparisonSlider');
const originalImg = document.getElementById('originalImg');
const enhancedImg = document.getElementById('enhancedImg');
const alertContainer = document.getElementById('alertContainer');
const histCanvas = document.getElementById('histChart');
const themeToggle = document.getElementById('themeToggle');
const thumbnailsContainer = document.getElementById('thumbnailsContainer');

// Theme Toggle
themeToggle.addEventListener('click', () => {
    document.body.classList.toggle('dark-theme');
    const isDark = document.body.classList.contains('dark-theme');
    themeToggle.textContent = isDark ? 'Light' : 'Dark';
    if (histChart) {
        plotHistogram(currentIndex);
    }
});

// Upload Area Click
uploadArea.addEventListener('click', () => fileInput.click());

// Drag and Drop
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.style.borderColor = '#999';
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.style.borderColor = '#ccc';
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.style.borderColor = '#ccc';
    const files = Array.from(e.dataTransfer.files).filter(f => f.type.startsWith('image/'));
    if (files.length > 0) handleFileUpload(files);
});

// File Input Change
fileInput.addEventListener('change', (e) => {
    const files = Array.from(e.target.files).filter(f => f.type.startsWith('image/'));
    if (files.length > 0) handleFileUpload(files);
});

// Handle Multiple File Upload
async function handleFileUpload(files) {
    // Filter valid files
    const validFiles = files.filter(file => {
        const validTypes = ['image/png', 'image/jpeg', 'image/jpg', 'image/bmp', 'image/tiff'];
        const type = file.type || '';
        if (!validTypes.some(t => type.includes(t.split('/')[1]))) {
            showAlert(`Invalid type for ${file.name}: PNG, JPG, JPEG, BMP, or TIFF only.`, 'error');
            return false;
        }
        if (file.size > 16 * 1024 * 1024) {
            showAlert(`${file.name} too large. Max 16MB.`, 'error');
            return false;
        }
        return true;
    });

    if (validFiles.length === 0) return;

    showAlert(`Uploading ${validFiles.length} images...`, 'success');

    const formData = new FormData();
    validFiles.forEach(file => formData.append('file', file));

    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });

        console.log('Upload response status:', response.status);
        const data = await response.json();
        console.log('Upload response data:', data);

        if (response.ok && data.success) {
            filenames = data.filenames;
            currentIndex = 0;
            updatePreview();
            previewSection.style.display = 'block';
            showAlert(`${validFiles.length} images uploaded! Select an enhancement method.`, 'success');
        } else {
            const errorMsg = data.error || 'Upload failed (unknown error)';
            console.error('Upload error:', errorMsg);
            showAlert(errorMsg, 'error');
        }
    } catch (error) {
        console.error('Fetch error:', error);
        showAlert('Upload error: ' + error.message, 'error');
    }
}

function updatePreview() {
    if (filenames.length > 0) {
        previewImage.src = `/get_image/uploads/${filenames[currentIndex]}?t=${Date.now()}`;
        previewCounter.textContent = `Image ${currentIndex + 1} of ${filenames.length}`;
        previewImage.onerror = () => showAlert('Failed to load preview', 'error');
    }
    updateNavButtons(prevPreview, nextPreview, filenames.length > 1);
}

// Update nav button states
function updateNavButtons(prevBtn, nextBtn, hasMultiple) {
    prevBtn.disabled = !hasMultiple || currentIndex === 0;
    nextBtn.disabled = !hasMultiple || currentIndex === filenames.length - 1;
}

// Navigation for Preview
prevPreview.addEventListener('click', () => {
    if (filenames.length > 1 && currentIndex > 0) {
        currentIndex--;
        updatePreview();
    }
});

nextPreview.addEventListener('click', () => {
    if (filenames.length > 1 && currentIndex < filenames.length - 1) {
        currentIndex++;
        updatePreview();
    }
});

// Process Images
async function processImage(method) {
    if (filenames.length === 0) {
        showAlert('Please upload images first', 'error');
        return;
    }

    selectedMethod = method;
    previewSection.style.display = 'none';
    loader.style.display = 'block';
    resultsSection.style.display = 'none';

    try {
        const response = await fetch('/process', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                filenames: filenames,
                method: method
            })
        });

        const data = await response.json();

        loader.style.display = 'none';
        if (data.success) {
            outputFilenames = data.output_filenames;
            originalHists = data.original_hists;
            enhancedHists = data.enhanced_hists;
            currentIndex = 0;
            populateThumbnails();
            updateResult();
            resultsSection.style.display = 'block';
            previewSection.style.display = 'block'; // Keep upload visible
            showAlert(`Enhancement complete for ${filenames.length} images using ${method}!`, 'success');
        } else {
            showAlert(data.error || 'Processing failed', 'error');
            previewSection.style.display = 'block';
        }
    } catch (error) {
        console.error('Processing error:', error);
        loader.style.display = 'none';
        showAlert('Processing error: ' + error.message, 'error');
        previewSection.style.display = 'block';
    }
}

function populateThumbnails() {
    thumbnailsContainer.innerHTML = '';
    outputFilenames.forEach((fname, idx) => {
        const thumb = document.createElement('img');
        thumb.src = `/get_image/results/${fname}?t=${Date.now()}`;
        thumb.className = 'thumbnail';
        thumb.onclick = () => selectThumbnail(idx);
        thumbnailsContainer.appendChild(thumb);
    });
    selectThumbnail(0);
}

function selectThumbnail(index) {
    currentIndex = index;
    updateResult();
}

function updateResult() {
    if (outputFilenames.length > 0) {
        originalImg.src = `/get_image/uploads/${filenames[currentIndex]}?t=${Date.now()}`;
        enhancedImg.src = `/get_image/results/${outputFilenames[currentIndex]}?t=${Date.now()}`;
        comparisonSlider.value = 50;
        resultCounter.textContent = `Image ${currentIndex + 1} of ${outputFilenames.length}`;
        plotHistogram(currentIndex);
    }
    updateNavButtons(prevResult, nextResult, outputFilenames.length > 1);
}

// Navigation for Results
prevResult.addEventListener('click', () => {
    if (outputFilenames.length > 1 && currentIndex > 0) {
        currentIndex--;
        updateResult();
    }
});

nextResult.addEventListener('click', () => {
    if (outputFilenames.length > 1 && currentIndex < outputFilenames.length - 1) {
        currentIndex++;
        updateResult();
    }
});

// Plot Histogram with two distinct colors (blue for original, green for enhanced)
function plotHistogram(index) {
    const ctx = histCanvas.getContext('2d');
    if (histChart) histChart.destroy();
    if (index >= 0 && originalHists[index] && enhancedHists[index]) {
        const isDark = document.body.classList.contains('dark-theme');
        histChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: Array.from({length: 256}, (_, i) => i),
                datasets: [
                    {
                        label: 'Original',
                        data: originalHists[index],
                        backgroundColor: 'rgba(0, 123, 255, 0.6)', // Blue for original
                        borderColor: 'rgba(0, 123, 255, 1)',
                        borderWidth: 1
                    },
                    {
                        label: 'Enhanced',
                        data: enhancedHists[index],
                        backgroundColor: 'rgba(40, 167, 69, 0.6)', // Green for enhanced
                        borderColor: 'rgba(40, 167, 69, 1)',
                        borderWidth: 1
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: { ticks: { color: isDark ? '#aaa' : '#666' } },
                    y: { beginAtZero: true, ticks: { color: isDark ? '#aaa' : '#666' } }
                },
                plugins: {
                    legend: { labels: { color: isDark ? '#aaa' : '#666' } },
                    title: { display: false }
                }
            }
        });
    }
}

// Comparison Slider
comparisonSlider.addEventListener('input', (e) => {
    const value = e.target.value;
    enhancedImg.style.clipPath = `inset(0 ${100 - value}% 0 0)`;
});

// Download
function downloadImage() {
    if (outputFilenames[currentIndex]) {
        window.location.href = `/download/${outputFilenames[currentIndex]}`;
    }
}

// Reset
function resetApp() {
    filenames = [];
    outputFilenames = [];
    originalHists = [];
    enhancedHists = [];
    currentIndex = 0;
    selectedMethod = '';
    fileInput.value = '';
    previewSection.style.display = 'none';
    resultsSection.style.display = 'none';
    alertContainer.innerHTML = '';
    thumbnailsContainer.innerHTML = '';
    if (histChart) {
        histChart.destroy();
        histChart = null;
    }
    showAlert('Ready for new uploads!', 'success');
}

// Alert
function showAlert(message, type) {
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type}`;
    alertDiv.textContent = message;
    alertContainer.innerHTML = '';
    alertContainer.appendChild(alertDiv);
    setTimeout(() => alertDiv.remove(), 5000);
}