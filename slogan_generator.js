// This script handles the client-side logic for the AI Slogan Generator.

document.addEventListener('DOMContentLoaded', () => {
    // Get references to HTML elements
    const keywordsInput = document.getElementById('keywordsInput'); // Input field for keywords
    const numSlogansSlider = document.getElementById('numSlogansSlider'); // Slider for number of slogans
    const numSlogansValueSpan = document.getElementById('numSlogansValue'); // Span to display slider value
    const generateBtn = document.getElementById('generateBtn'); // Button to trigger slogan generation
    const sloganList = document.getElementById('sloganList'); // Unordered list to display generated slogans
    const loader = document.getElementById('loader'); // Loading spinner element
    const errorMessage = document.getElementById('errorMessage'); // Element to display error messages

    // Add an event listener to the number of slogans slider
    // This updates the displayed value next to the slider as the user drags it.
    if (numSlogansSlider && numSlogansValueSpan) {
        numSlogansSlider.addEventListener('input', () => {
            numSlogansValueSpan.textContent = numSlogansSlider.value;
        });
    }

    // Add an event listener to the "Generate Slogans" button
    if (generateBtn) {
        generateBtn.addEventListener('click', async () => {
            // Get the keywords entered by the user, trim whitespace
            const keywords = keywordsInput ? keywordsInput.value.trim() : '';
            
            // Ensure numSlogans is correctly parsed from the slider value.
            // If numSlogansSlider exists, use its value. Otherwise, default to 5.
            let numSlogans = 5; // Default value
            if (numSlogansSlider && numSlogansSlider.value) {
                const parsedValue = parseInt(numSlogansSlider.value, 10);
                if (!isNaN(parsedValue)) {
                    numSlogans = parsedValue;
                }
            }
            
            // Clear previous error messages and slogans
            if (errorMessage) errorMessage.style.display = 'none';
            if (sloganList) sloganList.innerHTML = '';

            if (!keywords) {
                if (errorMessage) {
                    errorMessage.textContent = 'Please enter keywords or a topic to generate slogans.';
                    errorMessage.style.display = 'block';
                }
                return;
            }

            // Show loader and disable button
            if (loader) loader.style.display = 'block';
            if (generateBtn) {
                generateBtn.disabled = true;
                generateBtn.textContent = 'Generating...';
            }

            try {
                // Make a POST request to your Flask backend API
                const response = await fetch('/api/generate_slogan', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ keywords, num_slogans: numSlogans }), // Send both keywords and num_slogans
                });

                const data = await response.json();

                if (!response.ok) {
                    // If the response is not OK (e.g., 400 or 500 status)
                    throw new Error(data.error || 'Failed to generate slogans.');
                }

                // Display generated slogans
                if (sloganList) {
                    if (data.slogans && data.slogans.length > 0) {
                        // Iterate with index to add numbering
                        data.slogans.forEach((slogan, index) => {
                            const li = document.createElement('li');
                            li.textContent = `${index + 1}. ${slogan}`; // Add numbering here
                            sloganList.appendChild(li);
                        });
                    } else {
                        const li = document.createElement('li');
                        li.textContent = "No slogans generated. Try different keywords.";
                        sloganList.appendChild(li);
                    }
                }

            } catch (error) {
                console.error('Error generating slogans:', error);
                if (errorMessage) {
                    errorMessage.textContent = `Error: ${error.message}`;
                    errorMessage.style.display = 'block';
                }
                if (sloganList) {
                    const li = document.createElement('li');
                    li.textContent = "An error occurred while generating slogans. Please try again.";
                    sloganList.appendChild(li);
                }
            } finally {
                // Hide loader and re-enable button
                if (loader) loader.style.display = 'none';
                if (generateBtn) {
                    generateBtn.disabled = false;
                    generateBtn.textContent = 'Generate Slogans';
                }
            }
        });
    }

    // --- FAQ Accordion Functionality ---
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
                    const otherIcon = currentlyActive.querySelector('.faq-question i'); // Get the other icon
                    if (otherAnswer) otherAnswer.style.maxHeight = "0";
                    // Only manipulate icon if it exists to avoid errors on pages without it
                    if (otherIcon) { 
                        otherIcon.classList.remove('fa-minus'); // Change icon to plus
                        otherIcon.classList.add('fa-plus');
                    }
                }

                // Toggle the 'active' class on the clicked item
                item.classList.toggle('active');

                // If the clicked item is now active
                if (item.classList.contains('active')) {
                    if (answer) {
                        // Set the max-height to the scrollHeight to reveal the content smoothly
                        answer.style.maxHeight = answer.scrollHeight + "px";
                    }
                    if (icon) {
                        // Change icon to minus
                        icon.classList.remove('fa-plus');
                        icon.classList.add('fa-minus');
                    }
                } else {
                    // If the clicked item is no longer active (closing)
                    if (answer) {
                        answer.style.maxHeight = "0"; // Collapse the answer
                    }
                    if (icon) {
                        // Change icon to plus
                        icon.classList.remove('fa-minus');
                        icon.classList.add('fa-plus');
                    }
                }
            });
        }
    });
    // --- End FAQ Accordion Functionality ---
});
