document.addEventListener('DOMContentLoaded', () => {
    const keywordsInput = document.getElementById('keywordsInput');
    const ideasOutput = document.getElementById('ideasOutput');
    const generateIdeasBtn = document.getElementById('generateIdeasBtn');
    const loader = document.getElementById('loader');
    const errorMessage = document.getElementById('errorMessage');

    // Generate Ideas button click handler
    if (generateIdeasBtn && keywordsInput && ideasOutput && loader && errorMessage) {
        generateIdeasBtn.addEventListener('click', async () => {
            const keywords = keywordsInput.value.trim();

            if (!keywords) {
                errorMessage.textContent = 'Please enter keywords or a topic to generate ideas.';
                errorMessage.style.display = 'block';
                ideasOutput.value = ''; // Clear any previous output
                return;
            }

            // Clear previous output and error, show loader
            ideasOutput.value = '';
            errorMessage.style.display = 'none';
            loader.style.display = 'block';
            generateIdeasBtn.disabled = true; // Disable button during processing

            try {
                const response = await fetch('/api/generate_content_ideas', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ keywords }),
                });

                const data = await response.json();

                if (!response.ok) {
                    throw new Error(data.error || 'Failed to generate content ideas.');
                }

                // Display the generated ideas
                if (ideasOutput) {
                    ideasOutput.value = data.content_ideas;
                }

            } catch (error) {
                console.error('Error generating content ideas:', error);
                if (errorMessage) {
                    errorMessage.textContent = `Error: ${error.message}`;
                    errorMessage.style.display = 'block';
                }
                if (ideasOutput) {
                    ideasOutput.value = 'An error occurred while generating ideas. Please try again.';
                }
            } finally {
                // Hide loader and re-enable button
                if (loader) {
                    loader.style.display = 'none';
                }
                if (generateIdeasBtn) {
                    generateIdeasBtn.disabled = false;
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
