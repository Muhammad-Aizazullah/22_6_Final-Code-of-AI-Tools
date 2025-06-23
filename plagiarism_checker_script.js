document.addEventListener('DOMContentLoaded', () => {
    const inputTextarea = document.getElementById('inputText');
    const inputCounter = document.getElementById('inputCounter');
    const checkPlagiarismBtn = document.getElementById('checkPlagiarismBtn');

    // Elements for circular progress bars (Plagiarism graphical result)
    const plagiarismPieChart = document.getElementById('plagiarismPieChart');
    const uniquePieChart = document.getElementById('uniquePieChart');
    const plagiarismPercentageSpan = document.getElementById('plagiarismPercentage');
    const uniquePercentageSpan = document.getElementById('uniquePercentage');

    // Element for AI detection result
    const aiDetectionResultDiv = document.getElementById('aiDetectionResult');
    
    // Element for displaying general error messages to the user
    const errorMessageDiv = document.getElementById('errorMessage');

    // Zaroori variables jo HTML elements ko reference karte hain
    const loader = document.getElementById('loader'); 
    const resultCharts = document.querySelector('.result-charts'); 
    
    // Function to update word count for input
    inputTextarea.addEventListener('input', () => {
        const text = inputTextarea.value;
        const words = text.trim().split(/\s+/).filter(word => word.length > 0).length;
        inputCounter.textContent = `${words} words`;
        resetUI(); // Reset UI whenever input changes
    });

    // Function to update a single pie chart visual and percentage
    function updatePieChart(element, percentage, color, labelSpan) {
        element.style.background = `conic-gradient(
            ${color} 0% ${percentage}%,
            #e0e0e0 ${percentage}% 100%
        )`;
        labelSpan.textContent = `${percentage}%`; // Update the percentage text inside the chart
    }

    // Function to display error messages in the dedicated error div
    function showErrorMessage(message) {
        if (message) { // Only show if there's a message
            errorMessageDiv.textContent = message;
            errorMessageDiv.style.display = 'block'; 
        } else {
            errorMessageDiv.textContent = '';
            errorMessageDiv.style.display = 'none'; // Hide if message is empty
        }
    }

    // Function to clear results and error messages and reset UI to initial state
    function resetUI() {
        showErrorMessage(''); // Error message ko khali karein aur hide karein
        
        // Reset pie charts to 0% and make them visible
        updatePieChart(plagiarismPieChart, 0, 'red', plagiarismPercentageSpan);
        updatePieChart(uniquePieChart, 0, 'green', uniquePercentageSpan);

        // Reset AI detection result and make it visible with desired initial red color
        aiDetectionResultDiv.textContent = 'Estimated AI Content: --';
        aiDetectionResultDiv.style.backgroundColor = 'transparent'; // Make background transparent if desired, or keep white/default
        aiDetectionResultDiv.style.color = 'red'; // Always red for "--"
        aiDetectionResultDiv.style.fontWeight = 'bold'; // Make it bold for visibility
        aiDetectionResultDiv.style.display = 'block'; // AI content div ko shuru mein visible karein

        // Result charts section ko bhi visible karein initial state mein
        if (resultCharts) { 
            resultCharts.style.display = 'block'; // Charts section ko shuru mein visible karein
        }
    }

    // Handle the "Analyze Text" button click
    checkPlagiarismBtn.addEventListener('click', async () => {
        const text = inputTextarea.value.trim();

        // Reset UI to default empty/0% visible state before new analysis
        resetUI(); 

        if (text.length === 0) {
            showErrorMessage('Please enter text to analyze.');
            return;
        }

        // Set loading states for UI elements
        checkPlagiarismBtn.disabled = true;
        checkPlagiarismBtn.textContent = 'Analyzing...';
        checkPlagiarismBtn.classList.add('loading'); 

        // Loader ko show karein
        if (loader) { 
            loader.style.display = 'block';
        }

        // AI detection result div ko processing state mein update karein
        aiDetectionResultDiv.textContent = 'Processing AI detection...';
        aiDetectionResultDiv.style.backgroundColor = '#e0f7fa'; 
        aiDetectionResultDiv.style.color = '#000';
        aiDetectionResultDiv.style.fontWeight = 'normal';
        // aiDetectionResultDiv.style.display will remain 'block' from resetUI

        try {
            const response = await fetch('/api/check_plagiarism_ai', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: text }),
            });

            const data = await response.json();

            if (!response.ok) {
                const errorMsg = data.error || `Server responded with status ${response.status}`;
                throw new Error(errorMsg);
            }

            // --- Process Plagiarism Results ---
            const plagiarismScore = parseFloat(data.plagiarism_percentage);
            const uniqueScore = parseFloat(data.unique_percentage);

            if (!isNaN(plagiarismScore) && !isNaN(uniqueScore)) {
                updatePieChart(plagiarismPieChart, Math.round(plagiarismScore), 'red', plagiarismPercentageSpan);
                updatePieChart(uniquePieChart, Math.round(uniqueScore), 'green', uniquePercentageSpan);
            } else {
                 showErrorMessage('Plagiarism results not valid.');
            }
            
            // --- Process AI Detection Results ---
            const aiPercentage = parseFloat(data.ai_detection_probability);

            if (!isNaN(aiPercentage)) {
                aiDetectionResultDiv.textContent = `Estimated AI Content: ${aiPercentage}%`;
                // Make all AI content percentages red, regardless of the value
                aiDetectionResultDiv.style.backgroundColor = 'transparent'; // Or any desired background
                aiDetectionResultDiv.style.color = 'red'; // Always red
                aiDetectionResultDiv.style.fontWeight = 'bold';
            } else {
                // If AI percentage is not a valid number, display "--" in red
                aiDetectionResultDiv.textContent = 'Estimated AI Content: --';
                aiDetectionResultDiv.style.backgroundColor = 'transparent';
                aiDetectionResultDiv.style.color = 'red';
                aiDetectionResultDiv.style.fontWeight = 'bold';
            }

        } catch (error) {
            console.error('An unhandled error occurred:', error);
            showErrorMessage(`An unexpected error occurred during analysis: ${error.message}`);
            
            // Reset charts and AI result on unhandled errors
            updatePieChart(plagiarismPieChart, 0, 'red', plagiarismPercentageSpan);
            updatePieChart(uniquePieChart, 0, 'green', uniquePercentageSpan); 
            aiDetectionResultDiv.textContent = 'Estimated AI Content: Error';
            aiDetectionResultDiv.style.backgroundColor = '#ef9a9a'; 
            aiDetectionResultDiv.style.color = 'white'; 

        } finally {
            checkPlagiarismBtn.disabled = false;
            checkPlagiarismBtn.textContent = 'Analyze Text';
            checkPlagiarismBtn.classList.remove('loading');
            if (loader) { 
                loader.style.display = 'none'; 
            }
        }
    });

    // Dummy function for updateWordCount if it's not defined elsewhere
    function updateWordCount(textarea, counter) {
        const text = textarea.value;
        const words = text.trim().split(/\s+/).filter(word => word.length > 0).length;
        counter.textContent = `${words} words`;
    }

    // Initial setup when the page loads
    updateWordCount(inputTextarea, inputCounter); 
    resetUI(); // Call reset UI function to initialize all UI elements to their default visible state

    // --- FAQ Accordion Functionality ---
    const faqItems = document.querySelectorAll('.faq-item');

    faqItems.forEach(item => {
        const question = item.querySelector('.faq-question');
        const answer = item.querySelector('.faq-answer');
        const icon = item.querySelector('.faq-question i'); 
        
        if (question) { 
            question.addEventListener('click', () => {
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