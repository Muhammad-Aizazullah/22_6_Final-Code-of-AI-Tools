document.addEventListener('DOMContentLoaded', () => {
    // Get references to HTML elements
    const recipientInput = document.getElementById('recipientInput');
    const subjectInput = document.getElementById('subjectInput');
    const purposeInput = document.getElementById('purposeInput');
    const generateEmailBtn = document.getElementById('generateEmailBtn');
    const emailOutput = document.getElementById('emailOutput');
    const loader = document.getElementById('loader');
    const errorMessage = document.getElementById('errorMessage');

    // Generate Email button click handler
    if (generateEmailBtn && recipientInput && subjectInput && purposeInput && emailOutput && loader && errorMessage) {
        generateEmailBtn.addEventListener('click', async () => {
            const recipient = recipientInput.value.trim();
            const subject = subjectInput.value.trim();
            const purpose = purposeInput.value.trim();

            // Basic validation
            if (!subject && !purpose) {
                errorMessage.textContent = 'Please provide at least a Subject or Email Purpose/Keywords.';
                errorMessage.style.display = 'block';
                emailOutput.value = ''; // Clear any previous output
                return;
            }

            // Clear previous output and error, show loader
            emailOutput.value = '';
            errorMessage.style.display = 'none';
            loader.style.display = 'block';
            generateEmailBtn.disabled = true; // Disable button during processing

            try {
                // Send a POST request to the backend
                const response = await fetch('/api/generate_email', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ subject, purpose, recipient }),
                });

                const data = await response.json();

                // Check for errors from the API
                if (!response.ok) {
                    throw new Error(data.error || 'Failed to generate email.');
                }

                // Display the generated email
                if (emailOutput) {
                    emailOutput.value = data.generated_email;
                }

            } catch (error) {
                console.error('Error generating email:', error);
                if (errorMessage) {
                    errorMessage.textContent = `Error: ${error.message}`;
                    errorMessage.style.display = 'block';
                }
                if (emailOutput) {
                    emailOutput.value = 'An error occurred while generating the email. Please try again.';
                }
            } finally {
                // Hide loader and re-enable button
                if (loader) {
                    loader.style.display = 'none';
                }
                if (generateEmailBtn) {
                    generateEmailBtn.disabled = false;
                }
            }
        });
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
