/**
 * Bird Species Identification - Minimal Frontend Script
 * Features:
 * - Drag & drop file upload
 * - Clipboard image paste (Ctrl+V and Paste Button)
 * - File selection preview
 * - Asynchronous AJAX inference with error handling
 * - Real-time species search filter
 */

document.addEventListener('DOMContentLoaded', () => {
  const dropZone = document.getElementById('dropZone');
  const fileInput = document.getElementById('birdFileInput');
  const uploadTrigger = document.getElementById('uploadTrigger');
  const clipboardTrigger = document.getElementById('clipboardTrigger');
  const previewContainer = document.getElementById('previewContainer');
  const previewImage = document.getElementById('previewImage');
  const removeImageBtn = document.getElementById('removeImageBtn');
  const classifyBtn = document.getElementById('classifyBtn');
  const loadingBox = document.getElementById('loadingBox');
  const resultBox = document.getElementById('resultBox');
  const resultBirdName = document.getElementById('resultBirdName');
  const resultConfidence = document.getElementById('resultConfidence');
  const resultBarFill = document.getElementById('resultBarFill');
  const candidatesList = document.getElementById('candidatesList');
  const resultGoogleLink = document.getElementById('resultGoogleLink');
  const classifyForm = document.getElementById('birdClassifyForm');

  let activeFile = null;

  // 1. Browse File Button
  if (uploadTrigger && fileInput) {
    uploadTrigger.addEventListener('click', (e) => {
      e.stopPropagation();
      fileInput.click();
    });
  }

  // 2. Drag & Drop Handlers
  if (dropZone && fileInput) {
    ['dragenter', 'dragover'].forEach(name => {
      dropZone.addEventListener(name, (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropZone.classList.add('dragover');
      });
    });

    ['dragleave', 'drop'].forEach(name => {
      dropZone.addEventListener(name, (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropZone.classList.remove('dragover');
      });
    });

    dropZone.addEventListener('drop', (e) => {
      const files = e.dataTransfer.files;
      if (files && files.length > 0) {
        handleImageFile(files[0]);
      }
    });

    fileInput.addEventListener('change', () => {
      if (fileInput.files && fileInput.files[0]) {
        handleImageFile(fileInput.files[0]);
      }
    });
  }

  // 3. Clipboard Image Paste (Ctrl+V)
  window.addEventListener('paste', (e) => {
    const clipboardData = e.clipboardData || window.clipboardData;
    if (!clipboardData) return;

    const items = clipboardData.items;
    if (!items) return;

    for (let i = 0; i < items.length; i++) {
      if (items[i].type.indexOf('image') !== -1) {
        const file = items[i].getAsFile();
        if (file) {
          handleImageFile(file);
          break;
        }
      }
    }
  });

  // Paste Button Click
  if (clipboardTrigger) {
    clipboardTrigger.addEventListener('click', async (e) => {
      e.stopPropagation();
      try {
        if (navigator.clipboard && navigator.clipboard.read) {
          const items = await navigator.clipboard.read();
          for (const item of items) {
            for (const type of item.types) {
              if (type.startsWith('image/')) {
                const blob = await item.getType(type);
                const file = new File([blob], 'clipboard_image.png', { type: type });
                handleImageFile(file);
                return;
              }
            }
          }
          alert('No image found on clipboard. Copy an image first, then press Ctrl+V or click Paste.');
        } else {
          alert('Press Ctrl+V anywhere on this page to paste your copied image.');
        }
      } catch (err) {
        alert('Please press Ctrl+V to paste your image directly.');
      }
    });
  }

  // Process and preview valid image file
  function handleImageFile(file) {
    if (!file.type.startsWith('image/')) {
      alert('Please select or paste an image file (JPG, PNG, WEBP).');
      return;
    }

    activeFile = file;

    const reader = new FileReader();
    reader.onload = (e) => {
      if (previewImage) previewImage.src = e.target.result;
      if (previewContainer) previewContainer.style.display = 'block';
      if (resultBox) resultBox.style.display = 'none';
      if (classifyBtn) {
        classifyBtn.disabled = false;
        classifyBtn.style.opacity = '1';
      }
    };
    reader.readAsDataURL(file);
  }

  // Clear / Reset
  if (removeImageBtn) {
    removeImageBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      activeFile = null;
      if (fileInput) fileInput.value = '';
      if (previewContainer) previewContainer.style.display = 'none';
      if (resultBox) resultBox.style.display = 'none';
      if (classifyBtn) classifyBtn.disabled = true;
    });
  }

  // Form Submission via AJAX
  if (classifyForm) {
    classifyForm.addEventListener('submit', async (e) => {
      e.preventDefault();

      if (!activeFile && fileInput && fileInput.files && fileInput.files[0]) {
        activeFile = fileInput.files[0];
      }

      if (!activeFile) {
        alert('Please upload, drag, or paste an image first.');
        return;
      }

      const formData = new FormData();
      formData.append('image', activeFile);

      const csrfToken = document.querySelector('[name=csrfmiddlewaretoken]')?.value || '';

      if (loadingBox) loadingBox.style.display = 'block';
      if (resultBox) resultBox.style.display = 'none';
      if (classifyBtn) classifyBtn.disabled = true;

      try {
        const response = await fetch('/classify/', {
          method: 'POST',
          headers: {
            'X-Requested-With': 'XMLHttpRequest',
            'X-CSRFToken': csrfToken
          },
          body: formData
        });

        if (!response.ok) {
          const errData = await response.json().catch(() => ({}));
          throw new Error(errData.error || `Server returned error (${response.status})`);
        }

        const data = await response.json();
        renderResult(data);
      } catch (err) {
        if (loadingBox) loadingBox.style.display = 'none';
        if (classifyBtn) classifyBtn.disabled = false;
        alert(`Classification error: ${err.message}`);
      }
    });
  }

  function renderResult(data) {
    if (loadingBox) loadingBox.style.display = 'none';
    if (classifyBtn) classifyBtn.disabled = false;

    const birdName = data.bird || 'Unknown';
    const prob = parseFloat(data.probability || 0);

    if (resultBirdName) resultBirdName.innerText = birdName;
    if (resultConfidence) resultConfidence.innerText = `${prob.toFixed(1)}%`;
    if (resultBarFill) resultBarFill.style.width = `${Math.min(prob, 100)}%`;

    if (resultGoogleLink) {
      resultGoogleLink.href = `https://www.google.com/search?q=${encodeURIComponent(birdName)}+bird`;
    }

    if (candidatesList) {
      candidatesList.innerHTML = '';
      const topCandidates = data.top_predictions || [];
      topCandidates.slice(0, 4).forEach((item) => {
        const chip = document.createElement('div');
        chip.className = 'candidate-chip';
        chip.innerHTML = `
          <span>${item.species}</span>
          <span class="candidate-chip-prob">${item.probability}%</span>
        `;
        candidatesList.appendChild(chip);
      });
    }

    if (resultBox) {
      resultBox.style.display = 'block';
      resultBox.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
  }

  // Live Filter for Species Directory
  const speciesSearchInput = document.getElementById('speciesSearchInput');
  if (speciesSearchInput) {
    speciesSearchInput.addEventListener('input', (e) => {
      const q = e.target.value.toLowerCase().trim();
      const cards = document.querySelectorAll('.species-item-card');
      cards.forEach(card => {
        const name = card.getAttribute('data-name')?.toLowerCase() || '';
        card.style.display = name.includes(q) ? '' : 'none';
      });
    });
  }
});
