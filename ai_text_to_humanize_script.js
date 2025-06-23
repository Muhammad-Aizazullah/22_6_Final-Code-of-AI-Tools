document.addEventListener('DOMContentLoaded', () => {
    const inputText = document.getElementById('inputText');
    const outputText = document.getElementById('outputText');
    const humanizeBtn = document.getElementById('humanizeBtn');
    const loader = document.getElementById('loader');
    const errorMessage = document.getElementById('errorMessage');
    const inputCounter = document.getElementById('inputCounter');
    const outputCounter = document.getElementById('outputCounter');
    const creativitySlider = document.getElementById('creativitySlider');
    const creativityValue = document.getElementById('creativityValue');

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
            outputText.value = '';
            updateWordCount(outputText, outputCounter);
            errorMessage.style.display = 'none';
        });
    }

    // Update creativity slider value display
    if (creativitySlider && creativityValue) {
        creativitySlider.addEventListener('input', () => {
            creativityValue.textContent = `${creativitySlider.value}%`;
        });
    }

    // Humanize button click handler
    if (humanizeBtn && inputText && outputText && loader && errorMessage && creativitySlider) {
        humanizeBtn.addEventListener('click', async () => {
            const text = inputText.value.trim();
            const creativity = parseInt(creativitySlider.value) / 100; // Convert to ratio 0.0-1.0

            if (!text) {
                errorMessage.textContent = 'Please enter text to humanize.';
                errorMessage.style.display = 'block';
                outputText.value = '';
                updateWordCount(outputText, outputCounter);
                return;
            }

            // Clear previous output and error, show loader
            outputText.value = '';
            updateWordCount(outputText, outputCounter);
            errorMessage.style.display = 'none';
            loader.style.display = 'block';
            humanizeBtn.disabled = true; // Disable button during processing

            try {
                const response = await fetch('/api/humanize', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ text, creativity }),
                });

                const data = await response.json();

                if (!response.ok) {
                    throw new Error(data.error || 'Failed to humanize text.');
                }

                outputText.value = data.humanized_text;
                updateWordCount(outputText, outputCounter);

            } catch (error) {
                console.error('Error humanizing text:', error);
                errorMessage.textContent = `Error: ${error.message}`;
                errorMessage.style.display = 'block';
                outputText.value = 'An error occurred during humanization. Please try again.';
                updateWordCount(outputText, outputCounter);
            } finally {
                loader.style.display = 'none';
                humanizeBtn.disabled = false;
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
