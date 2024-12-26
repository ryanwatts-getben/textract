// Custom JavaScript for Swagger UI enhancements

function expandAllModels() {
    const modelContainers = document.querySelectorAll('.model-container');
    modelContainers.forEach(container => {
        container.classList.add('is-open');
        const toggle = container.querySelector('.model-toggle');
        if (toggle) {
            toggle.setAttribute('aria-expanded', 'true');
            toggle.style.transform = 'rotate(90deg)';
        }
    });
}

// Apply enhancements periodically to catch dynamically loaded elements
function applyEnhancements() {
    expandAllModels();
}

// Initial application
document.addEventListener('DOMContentLoaded', () => {
    // Initial application
    applyEnhancements();

    // Reapply periodically for dynamic content
    setInterval(applyEnhancements, 1000);

    // Apply on any click (for dynamic content)
    document.addEventListener('click', () => {
        setTimeout(applyEnhancements, 100);
    });
}); 