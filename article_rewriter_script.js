// --- Article Rewriter Tool JS ---
const inputText = document.getElementById('inputText');
const outputText = document.getElementById('outputText');
const rewriteBtn = document.getElementById('rewriteBtn');
const lengthSlider = document.getElementById('lengthSlider');
const sliderValue = document.getElementById('sliderValue');
const inputCounter = document.getElementById('inputCounter');
const outputCounter = document.getElementById('outputCounter');
const errorMessage = document.getElementById('errorMessage'); // Added error message element
const loader = document.getElementById('loader'); // Added loader element

// Update slider value display
if (lengthSlider && sliderValue) {
    lengthSlider.addEventListener('input', () => {
        sliderValue.textContent = `${lengthSlider.value}%`;
    });
}

// Function to update word count for a given textarea
function updateWordCount(textArea, counterElement) {
    if (!textArea || !counterElement) return;
    const text = textArea.value;
    if (text.trim() === '') {
        counterElement.textContent = '0 words';
    } else {
        const wordCount = text.trim().split(/\s+/).filter(word => word.length > 0).length;
        counterElement.textContent = `${wordCount} words`;
    }
}

// Update word count for input text area on load and input
if (inputText && inputCounter) {
    inputText.addEventListener('input', () => {
        updateWordCount(inputText, inputCounter);
        // Clear output and hide error/loader when input changes
        outputText.value = ''; // Clear output text
        updateWordCount(outputText, outputCounter); // Reset output word count
        errorMessage.style.display = 'none'; // Hide any previous error messages
        loader.style.display = 'none'; // Hide loader if it was active
    });
    // Initial update on page load
    updateWordCount(inputText, inputCounter);
}

// Rewrite button click handler
if (rewriteBtn && inputText && outputText && lengthSlider && errorMessage && loader) {
    rewriteBtn.addEventListener('click', async () => {
        const text = inputText.value.trim();
        const creativity_ratio = parseInt(lengthSlider.value) / 100; // Convert slider value to a 0-1 ratio

        // Clear previous outputs and error messages
        outputText.value = '';
        updateWordCount(outputText, outputCounter);
        errorMessage.style.display = 'none';

        if (!text) {
            errorMessage.textContent = 'Please enter an article to rewrite.';
            errorMessage.style.display = 'block';
            return;
        }

        // Show loader and disable button
        loader.style.display = 'block';
        rewriteBtn.disabled = true;
        rewriteBtn.textContent = 'Rewriting...'; // Change button text
        rewriteBtn.classList.add('loading'); // Add a loading class for styling

        try {
            const response = await fetch('/api/rewrite', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ text, creativity: creativity_ratio }),
            });

            const data = await response.json();

            if (!response.ok) {
                // If response is not OK (e.g., 400, 500 status)
                throw new Error(data.error || 'Failed to rewrite article.');
            }

            outputText.value = data.rewritten_text;

        } catch (error) {
            console.error('Error rewriting article:', error);
            errorMessage.textContent = `Error: ${error.message}`;
            errorMessage.style.display = 'block';
            outputText.value = 'An error occurred during rewriting. Please try again.';
        } finally {
            // Hide loader and re-enable button regardless of success or failure
            loader.style.display = 'none';
            rewriteBtn.disabled = false;
            rewriteBtn.textContent = 'Rewrite Article';
            rewriteBtn.classList.remove('loading');
            updateWordCount(outputText, outputCounter); // Update word count for output
        }
    });
}

// Initial word count update for output text area (will be 0 initially)
if (outputText && outputCounter) {
    updateWordCount(outputText, outputCounter);
}

// --- FAQ Accordion JS ---
const faqItems = document.querySelectorAll('.faq-item');
faqItems.forEach(item => {
    const question = item.querySelector('.faq-question');
    const answer = item.querySelector('.faq-answer');
    const icon = item.querySelector('.faq-question i'); // Get the icon element

    if (question) {
        question.addEventListener('click', () => {
            // Close other active items
            const currentlyActive = document.querySelector('.faq-item.active');
            if (currentlyActive && currentlyActive !== item) {
                currentlyActive.classList.remove('active');
                const otherAnswer = currentlyActive.querySelector('.faq-answer');
                const otherIcon = currentlyActive.querySelector('.faq-question i');
                if (otherAnswer) otherAnswer.style.maxHeight = "0";
                if (otherIcon) { // Ensure icon exists before manipulating
                    otherIcon.classList.remove('fa-minus');
                    otherIcon.classList.add('fa-plus');
                }
            }
            // Toggle current item
            item.classList.toggle('active');

            if (item.classList.contains('active')) {
                if (answer) {
                    answer.style.maxHeight = answer.scrollHeight + "px";
                }
                if (icon) { // Ensure icon exists before manipulating
                    icon.classList.remove('fa-plus');
                    icon.classList.add('fa-minus');
                }
            } else {
                if (answer) {
                    answer.style.maxHeight = "0";
                }
                if (icon) { // Ensure icon exists before manipulating
                    icon.classList.remove('fa-minus');
                    icon.classList.add('fa-plus');
                }
            }
        });
    }
});
