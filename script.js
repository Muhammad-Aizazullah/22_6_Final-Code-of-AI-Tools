// --- Summarizer Tool JS ---
const inputText = document.getElementById('inputText');
const outputText = document.getElementById('outputText');
const summarizeBtn = document.getElementById('summarizeBtn');
const lengthSlider = document.getElementById('lengthSlider');
const sliderValue = document.getElementById('sliderValue');
const inputCounter = document.getElementById('inputCounter');
const outputCounter = document.getElementById('outputCounter');

// Update slider value display
if (lengthSlider && sliderValue) { // Ensure elements exist
    lengthSlider.addEventListener('input', () => {
        sliderValue.textContent = `${lengthSlider.value}%`;
    });
}

// Function to update word count for a given textarea
function updateWordCount(textArea, counterElement) {
    if (!textArea || !counterElement) return; // Defensive check
    const text = textArea.value;
    if (text.trim() === '') {
        counterElement.textContent = '0 words';
    } else {
        // Split by whitespace to count words
        const wordCount = text.trim().split(/\s+/).filter(word => word.length > 0).length; // Filter empty strings
        counterElement.textContent = `${wordCount} words`;
    }
}

// Update word count for input text area
if (inputText && inputCounter) { // Ensure elements exist
    inputText.addEventListener('input', () => {
        updateWordCount(inputText, inputCounter);
        // Clear output and hide error/loader when input changes
        if (outputText) outputText.value = '';
        if (outputCounter) updateWordCount(outputText, outputCounter);
    });
}

// Summarize button click handler
if (summarizeBtn && inputText && outputText && lengthSlider) { // Ensure all necessary elements exist
    summarizeBtn.addEventListener('click', async () => {
        const text = inputText.value.trim();
        const length = lengthSlider.value; // Get the percentage from the slider

        // Basic validation
        if (!text) {
            // In a real application, you might want to show an alert or error message here.
            console.error('Please enter text to summarize.');
            return;
        }

        // Disable button and show a loading indicator (if you have one)
        summarizeBtn.disabled = true;
        summarizeBtn.textContent = 'Summarizing...'; // Provide feedback to the user
        // You might also add a visual loader/spinner here

        try {
            // Send the text to your Flask backend for summarization
            const response = await fetch('/api/summarize', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ text, length }), // Send text and desired length
            });

            const data = await response.json();

            if (!response.ok) {
                // Handle API errors (e.g., if Flask returns an error status)
                throw new Error(data.error || 'Failed to summarize text.');
            }

            // Display the summarized text
            if (outputText) {
                outputText.value = data.summary;
                updateWordCount(outputText, outputCounter); // Update word count for output
            }

        } catch (error) {
            console.error('Error during summarization:', error);
            if (outputText) {
                outputText.value = 'Error: Could not summarize text. Please try again.';
            }
            // You might want to display this error in a more user-friendly way on the page
        } finally {
            // Re-enable button and reset text
            summarizeBtn.disabled = false;
            summarizeBtn.textContent = 'Summarize';
            // Hide loading indicator
        }
    });
}

// Initial update on page load for both input and output text areas
if (inputText && inputCounter) {
    updateWordCount(inputText, inputCounter);
}
if (outputText && outputCounter) { // Also update for output, initially 0
    updateWordCount(outputText, outputCounter);
}


// --- FAQ Accordion JS ---
const faqItems = document.querySelectorAll('.faq-item');
faqItems.forEach(item => {
    const question = item.querySelector('.faq-question');
    const answer = item.querySelector('.faq-answer');
    const icon = item.querySelector('.faq-question i'); // Get the icon element

    if (question) { // Ensure question element exists
        question.addEventListener('click', () => {
            // Close other active items
            const currentlyActive = document.querySelector('.faq-item.active');
            if (currentlyActive && currentlyActive !== item) {
                currentlyActive.classList.remove('active');
                const otherAnswer = currentlyActive.querySelector('.faq-answer');
                const otherIcon = currentlyActive.querySelector('.faq-question i'); // Get the other icon
                if (otherAnswer) otherAnswer.style.maxHeight = "0";
                if (otherIcon) { // Ensure otherIcon exists before manipulating
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
