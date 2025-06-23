// This script handles the client-side logic for the AI Paraphrasing Tool.

document.addEventListener('DOMContentLoaded', () => {
    // Get references to HTML elements
    const inputText = document.getElementById('inputText');
    const outputText = document.getElementById('outputText');
    const paraphraseBtn = document.getElementById('paraphraseBtn');
    const loader = document.getElementById('loader');
    const errorMessage = document.getElementById('errorMessage');
    const inputCounter = document.getElementById('inputCounter');
    const outputCounter = document.getElementById('outputCounter');

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

    // Initial update on page load for input text area
    if (inputText && inputCounter) {
        updateWordCount(inputText, inputCounter);
        // Update word count for input text area on input
        inputText.addEventListener('input', () => {
            updateWordCount(inputText, inputCounter);
            // Clear output and hide error/loader when input changes
            if (outputText) {
                outputText.value = '';
                updateWordCount(outputText, outputCounter);
            }
            if (errorMessage) errorMessage.style.display = 'none';
        });
    }

    // Paraphrase button click handler
    if (paraphraseBtn && inputText && outputText && loader && errorMessage) {
        paraphraseBtn.addEventListener('click', async () => {
            const text = inputText.value.trim();

            if (!text) {
                errorMessage.textContent = 'Please enter text to paraphrase.';
                errorMessage.style.display = 'block';
                if (outputText) {
                    outputText.value = '';
                    updateWordCount(outputText, outputCounter);
                }
                return;
            }

            // Clear previous output and error, show loader
            if (outputText) {
                outputText.value = '';
                updateWordCount(outputText, outputCounter);
            }
            errorMessage.style.display = 'none';
            loader.style.display = 'block';
            paraphraseBtn.disabled = true; // Disable button during processing
            paraphraseBtn.textContent = 'Paraphrasing...';

            try {
                // Make a POST request to your Flask backend API
                // This URL should be '/api/rewrite' as defined in app.py
                const response = await fetch('/api/rewrite', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ text }),
                });

                const data = await response.json();

                if (!response.ok) {
                    // If the response is not OK (e.g., 400 or 500 status)
                    throw new Error(data.error || 'Failed to paraphrase text.');
                }

                // Expecting 'paraphrased_text' key from the backend's JSON response
                if (data.rewritten_text !== undefined) { // Check for the correct key
                    outputText.value = data.rewritten_text; // Use rewritten_text key
                } else {
                    outputText.value = 'Error: Received unexpected data format from server.';
                    errorMessage.textContent = 'Server response missing "rewritten_text".';
                    errorMessage.style.display = 'block';
                }
                updateWordCount(outputText, outputCounter);

            } catch (error) {
                console.error('Error paraphrasing text:', error);
                errorMessage.textContent = `Error: ${error.message}`;
                errorMessage.style.display = 'block';
                if (outputText) {
                    outputText.value = 'An error occurred during paraphrasing. Please try again.';
                    updateWordCount(outputText, outputCounter);
                }
            } finally {
                // Hide loader and re-enable button
                loader.style.display = 'none';
                paraphraseBtn.disabled = false;
                paraphraseBtn.textContent = 'Paraphrase Text';
            }
        });
    }

    // Initial word count update for output text area (will be 0 initially)
    if (outputText && outputCounter) {
        updateWordCount(outputText, outputCounter);
    }

    // --- FAQ Accordion Functionality ---
    const faqItems = document.querySelectorAll('.faq-item');

    faqItems.forEach(item => {
        const question = item.querySelector('.faq-question');
        const answer = item.querySelector('.faq-answer');
        const icon = item.querySelector('.faq-question i');

        if (question) {
            question.addEventListener('click', () => {
                // Close other active items
                const currentlyActive = document.querySelector('.faq-item.active');
                if (currentlyActive && currentlyActive !== item) {
                    currentlyActive.classList.remove('active');
                    const otherAnswer = currentlyActive.querySelector('.faq-answer');
                    const otherIcon = currentlyActive.querySelector('.faq-question i');
                    if (otherAnswer) otherAnswer.style.maxHeight = "0";
                    if (otherIcon) {
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
                    if (icon) {
                        icon.classList.remove('fa-plus');
                        icon.classList.add('fa-minus');
                    }
                } else {
                    if (answer) {
                        answer.style.maxHeight = "0";
                    }
                    if (icon) {
                        icon.classList.remove('fa-minus');
                        icon.classList.add('fa-plus');
                    }
                }
            });
        }
    });
    // --- End FAQ Accordion Functionality ---
});
