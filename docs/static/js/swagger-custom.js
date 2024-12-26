window.onload = function () {
    // Wait until the Swagger UI fully renders
    setTimeout(() => {
        // Select all copy buttons and force visibility
        document.querySelectorAll('.copy-to-clipboard').forEach(button => {
            button.style.display = 'inline-flex';
            button.style.opacity = '1';
            button.style.visibility = 'visible';
            button.style.position = 'relative';
        });

        // Prevent future hiding by listening for DOM changes
        const observer = new MutationObserver(() => {
            document.querySelectorAll('.copy-to-clipboard').forEach(button => {
                button.style.display = 'inline-flex';
                button.style.opacity = '1';
                button.style.visibility = 'visible';
            });
        });

        observer.observe(document.body, {
            childList: true,
            subtree: true
        });
    }, 500);
};
