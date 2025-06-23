import nltk
import os
import logging
import re
import sys 
import random # For simulating AI detection and plagiarism
import json # Added for parsing Gemini's JSON response

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, T5Tokenizer
import torch

# IMPORTANT: For Gemini API integration, you need to install the google-generativeai library.
# Run 'pip install google-generativeai' in your virtual environment.
try:
    import google.generativeai as genai
    # Configure your Gemini API key here.
    # Replace "YOUR_GEMINI_API_KEY" with your actual API key from Google AI Studio.
    # This key is crucial for the Gemini model to work.
    # ****** You MUST replace "YOUR_GEMINI_API_KEY" with your valid API key ******
    genai.configure(api_key="AIzaSyBMPRxeEJ63IBPYPEsBsJ_ShMWcI7d0h4A") 
    logging.info("Google Generative AI library loaded and configured.")
    GEMINI_API_AVAILABLE = True
except ImportError:
    logging.warning("Google Generative AI library not found. Gemini functions will not work.")
    GEMINI_API_AVAILABLE = False
except Exception as e:
    logging.error(f"Error configuring Gemini API: {e}. Gemini functions might not work.")
    GEMINI_API_AVAILABLE = False


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Set NLTK Data Path ---
nltk_data_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'nltk_data')
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)
    logging.info(f"Created NLTK data directory: {nltk_data_dir}")
nltk.data.path.append(nltk_data_dir)
logging.info(f"NLTK data path added: {nltk_data_dir}")
# --- End NLTK Data Path Setup ---

# --- NLTK Data Check ---
def ensure_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
        logging.info("NLTK 'punkt' tokenizer already exists.")
    except nltk.downloader.DownloadError:
        logging.info("Downloading NLTK 'punkt' tokenizer...")
        nltk.download('punkt', download_dir=nltk_data_dir)
        logging.info("NLTK 'punkt' tokenizer downloaded.")
    
    try:
        nltk.data.find('corpora/stopwords')
        logging.info("NLTK 'stopwords' corpus already exists.")
    except nltk.downloader.DownloadError:
        logging.info("Downloading NLTK 'stopwords' corpus...")
        nltk.download('stopwords', download_dir=nltk_data_dir)
        logging.info("NLTK 'stopwords' corpus downloaded.")

# Ensure NLTK data is available when the app starts
ensure_nltk_data()

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app) # Enable CORS for all routes

# --- Global variables for models and tokenizers ---
# Initialize to None and load lazily or in a pre-run setup
summarizer_tokenizer = None
summarizer_model = None
paraphraser_tokenizer = None
paraphraser_model = None
# GPT-2 models for humanizer, email generator, content idea generator will be replaced by Gemini.
# humanizer_tokenizer = None
# humanizer_model = None
# email_generator_tokenizer = None
# email_generator_model = None
# content_idea_generator_tokenizer = None
# content_idea_generator_model = None


# Determine device for PyTorch (CPU or GPU if available)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"Using device: {DEVICE}")

# --- Model Loading Functions (Lazy Loading) ---
# Only T5 models are explicitly loaded as others will use Gemini API
def load_summarizer_model():
    """Loads the T5-small model and tokenizer for summarization."""
    global summarizer_tokenizer, summarizer_model
    if summarizer_model is None:
        try:
            logging.info("Loading summarizer model (t5-small)...")
            summarizer_tokenizer = T5Tokenizer.from_pretrained("t5-small")
            summarizer_model = AutoModelForSeq2SeqLM.from_pretrained("t5-small").to(DEVICE)
            logging.info("Summarizer model loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading summarizer model: {e}")
            summarizer_tokenizer = None
            summarizer_model = None
    return summarizer_tokenizer, summarizer_model

def load_paraphraser_model():
    """Loads a T5-small model and tokenizer for paraphrasing/rewriting."""
    global paraphraser_tokenizer, paraphraser_model
    if paraphraser_model is None:
        try:
            logging.info("Loading paraphraser model (t5-small)...")
            paraphraser_tokenizer = T5Tokenizer.from_pretrained("t5-small")
            paraphraser_model = AutoModelForSeq2SeqLM.from_pretrained("t5-small").to(DEVICE)
            logging.info("Paraphraser model loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading paraphraser model: {e}")
            paraphraser_tokenizer = None
            paraphraser_model = None
    return paraphraser_tokenizer, paraphraser_model

# --- Core Logic Functions ---

def summarize_text(text, max_length_ratio=0.5):
    """Summarizes the given text using the loaded summarizer model."""
    tokenizer, model = load_summarizer_model()
    if model is None:
        return "Error: Summarizer model not loaded."
    
    input_ids = tokenizer.encode(text, return_tensors="pt").to(DEVICE)
    input_length = input_ids.shape[1]
    
    min_length = max(20, int(input_length * (max_length_ratio - 0.2)))
    max_length = max(min_length + 10, int(input_length * max_length_ratio))
    
    summary_ids = model.generate(
        input_ids,
        max_length=max_length,
        min_length=min_length,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def rewrite_article(text, creativity=0.5):
    """Rewrites/paraphrases the given text using the loaded paraphraser model."""
    tokenizer, model = load_paraphraser_model()
    if model is None:
        # Changed error message to be consistent with API response
        return "Error: Paraphraser model not loaded."

    input_text = text
    tokenized_text = tokenizer.encode(input_text, return_tensors="pt").to(DEVICE)

    temperature = 0.5 + (creativity * 0.5) 

    output_ids = model.generate(
        tokenized_text,
        max_length=int(tokenized_text.shape[1] * 1.2) + 50,
        min_length=int(tokenized_text.shape[1] * 0.8),
        num_beams=5,
        no_repeat_ngram_size=2,
        early_stopping=True,
        do_sample=True,
        temperature=temperature,
        top_k=50,
        top_p=0.95
    )
    rewritten_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    rewritten_text = re.sub(r'\s*([.,;!?])', r'\1', rewritten_text)
    rewritten_text = re.sub(r'\n+', '\n', rewritten_text).strip()

    return rewritten_text


def humanize_text_content(text, creativity_level=0.7):
    """Humanizes AI-generated text using the Gemini model."""
    if not GEMINI_API_AVAILABLE:
        logging.error("Gemini API is not available for humanization. Please install 'google-generativeai' and configure API key.")
        return f"Error: Gemini API not configured for humanization. Please install 'google-generativeai' and set your API key in app.py."

    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        prompt = (
            f"Please rewrite the following text to sound more natural, human-like, and engaging. "
            f"Aim for a {int(creativity_level*100)}% creative flair while maintaining the original meaning. "
            "Remove any robotic or overly formal phrasing, and inject a natural flow.\n\n"
            f"Original text:\n---\n{text}\n---\n\nRewritten human-like version:"
        )
        
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.9, # High temperature for creativity
                top_p=0.9,
                top_k=40,
                candidate_count=1,
            )
        )
        
        if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
            humanized_text = response.candidates[0].content.parts[0].text.strip()
            # Post-processing to remove any leading/trailing prompt phrases Gemini might generate
            humanized_text = re.sub(r'^(Rewritten human-like version:)?\s*', '', humanized_text, flags=re.IGNORECASE).strip()
            return humanized_text
        else:
            return "No humanized text could be generated by Gemini. Try different input."

    except Exception as e:
        logging.error(f"Error humanizing text with Gemini: {e}", exc_info=True) # Added exc_info=True
        return f"Error: Humanization failed with Gemini. Details: {str(e)}"


def generate_email_content(subject, purpose, recipient=''):
    """Generates email content using the Gemini model."""
    if not GEMINI_API_AVAILABLE:
        logging.error("Gemini API is not available for email generation. Please install 'google-generativeai' and configure API key.")
        return f"Error: Gemini API not configured for email generation. Please install 'google-generativeai' and set your API key in app.py."

    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        email_prompt = (
            f"Generate a professional email.\n"
            f"Subject: {subject}\n"
            f"Purpose: {purpose}\n"
        )
        if recipient:
            email_prompt += f"Recipient: {recipient}\n"
        email_prompt += "\nFormat the email appropriately, starting with 'Dear [Recipient Name or Team],' and ending with a professional closing."

        response = model.generate_content(
            email_prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.8, # Good balance for professional tone and some variability
                top_p=0.8,
                top_k=30,
                candidate_count=1,
            )
        )
        
        if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
            generated_email = response.candidates[0].content.parts[0].text.strip()
            return generated_email
        else:
            return "No email content could be generated by Gemini. Try different details."

    except Exception as e:
        logging.error(f"Error generating email with Gemini: {e}", exc_info=True) # Added exc_info=True
        return f"Error: Email generation failed with Gemini. Details: {str(e)}"


def generate_content_ideas(keywords):
    """Generates content ideas based on keywords using the Gemini model."""
    if not GEMINI_API_AVAILABLE:
        logging.error("Gemini API is not available for content idea generation. Please install 'google-generativeai' and configure API key.")
        return "Error: Gemini API not configured for content idea generation. Please install 'google-generativeai' and set your API key in app.py."

    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        # Request a structured JSON output for content ideas
        response_schema = {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "idea": {"type": "STRING"}
                }
            }
        }

        prompt = (
            f"Generate 7 creative, unique, and engaging content ideas related to '{keywords}'. "
            "Focus on diverse angles, trending topics, and actionable ideas. "
            "Each idea should be concise and compelling."
        )

        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                response_mime_type="application/json",
                response_schema=response_schema,
                temperature=0.9, # High temperature for creativity
                top_p=0.9,
                top_k=40,
                candidate_count=1,
            )
        )
        
        if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
            json_string = response.candidates[0].content.parts[0].text
            ideas_data = json.loads(json_string)
            
            ideas_list = [item["idea"].strip() for item in ideas_data if "idea" in item]
            
            # Format them consistently as a numbered list
            formatted_ideas = []
            for i, idea in enumerate(ideas_list):
                if idea:
                    formatted_ideas.append(f"{i + 1}. {idea}")

            if not formatted_ideas:
                return "No specific ideas generated by Gemini. Try different keywords or consider:\n1. Introduction to your field\n2. Common challenges\n3. Future trends\n4. How-to guides"
            
            return "\n".join(formatted_ideas)
        else:
            return "No content ideas could be generated by Gemini. Try different keywords."

    except Exception as e:
        logging.error(f"Error generating content ideas with Gemini: {e}", exc_info=True) # Added exc_info=True
        return f"Error: Content idea generation failed with Gemini. Details: {str(e)}"


def generate_slogans(keywords, num_slogans=5):
    """Generates slogans using the Gemini model."""
    if not GEMINI_API_AVAILABLE:
        logging.error("Gemini API is not available. Please install 'google-generativeai' and configure API key.")
        return [f"Error: Gemini API not configured. Please install 'google-generativeai' and set your API key in app.py."]

    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        # Define the structure for the desired JSON output from the model
        response_schema = {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "slogan": {"type": "STRING"}
                }
            }
        }

        # Craft the prompt for Gemini. Requesting numbered list explicitly.
        prompt = (
            f"Generate {num_slogans} unique, catchy, and memorable advertising slogans "
            f"for a brand or campaign related to '{keywords}'. "
            "Each slogan should be concise and highly impactful."
        )

        # Generate content with structured output
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                response_mime_type="application/json",
                response_schema=response_schema,
                temperature=0.9,  # Higher temperature for creativity
                top_p=0.9,
                top_k=40,
                candidate_count=1,
            )
        )
        
        # Parse the structured JSON response
        if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
            # The response.text will contain the JSON string
            json_string = response.candidates[0].content.parts[0].text
            slogans_data = json.loads(json_string) 
            
            slogans_list = [item["slogan"].strip() for item in slogans_data if "slogan" in item]
            
            # Ensure we return exactly num_slogans if possible, or as many as generated
            unique_slogans = []
            seen_slogans = set()
            for slogan in slogans_list:
                if slogan.lower() not in seen_slogans:
                    unique_slogans.append(slogan)
                    seen_slogans.add(slogan.lower())
                if len(unique_slogans) >= num_slogans:
                    break
            
            if not unique_slogans:
                return ["No slogans could be generated by Gemini. Try different keywords."]
            
            return unique_slogans[:num_slogans]

    except Exception as e:
        logging.error(f"Error generating slogans with Gemini: {e}", exc_info=True) # Added exc_info=True
        # Add more specific error messages if needed, e.g., API key invalid, quota exceeded
        return [f"Error: Slogan generation failed with Gemini. Details: {str(e)}"]


def check_plagiarism_and_ai(text):
    """
    Simulates plagiarism and AI detection.
    
    NOTE: This is a simplified, placeholder implementation.
    A real plagiarism checker would involve comparing the text against a vast
    database of documents (e.g., via web scraping or a pre-indexed corpus)
    and complex similarity algorithms.
    A real AI detector would use sophisticated classification models.
    """
    logging.info("Simulating plagiarism and AI detection...")

    # --- Simulated Plagiarism Check ---
    sentences = nltk.sent_tokenize(text)
    
    if len(sentences) < 2:
        plagiarism_percentage = random.randint(0, 50)
        unique_percentage = 100 - plagiarism_percentage
    else:
        source_text = " ".join(sentences[:len(sentences)//2])
        
        vectorizer = TfidfVectorizer().fit([text, source_text])
        vectors = vectorizer.transform([text, source_text])
        
        similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
        
        plagiarism_percentage = int(similarity * 100)
        unique_percentage = 100 - plagiarism_percentage

        plagiarism_percentage = max(0, min(100, plagiarism_percentage + random.randint(-15, 15)))
        unique_percentage = 100 - plagiarism_percentage

    # --- Simulated AI Detection ---
    ai_detection_probability = random.randint(10, 95) # Random probability between 10% and 95%

    logging.info(f"Simulated Plagiarism: {plagiarism_percentage}%, Unique: {unique_percentage}%, AI Probability: {ai_detection_probability}%")
    
    return {
        "plagiarism_percentage": plagiarism_percentage,
        "unique_percentage": unique_percentage,
        "ai_detection_probability": ai_detection_probability
    }

# --- Flask Routes for HTML Pages ---

@app.route('/')
def index():
    logging.info("Serving index.html")
    return render_template('index.html')

@app.route('/article-rewriter')
def article_rewriter_page():
    logging.info("Serving article_rewriter.html")
    return render_template('article_rewriter.html')

@app.route('/plagiarism-checker')
def plagiarism_checker_page():
    logging.info("Serving plagiarism_checker.html")
    return render_template('plagiarism_checker.html')

@app.route('/paraphraser')
def paraphraser_page():
    logging.info("Serving paraphrasing_tool.html")
    return render_template('paraphrasing_tool.html')

@app.route('/content_ideas')
def content_ideas_page():
    logging.info("Serving content_idea_generator.html")
    return render_template('content_idea_generator.html')

@app.route('/slogan_generator')
def slogan_generator_page():
    logging.info("Serving slogan_generator.html")
    return render_template('slogan_generator.html')

@app.route('/ai_humanizer')
def ai_humanizer_page():
    logging.info("Serving ai_text_to_humanize.html")
    return render_template('ai_text_to_humanize.html')

@app.route('/ai_email_generator')
def ai_email_generator_page():
    logging.info("Serving ai_email_generator.html")
    return render_template('ai_email_generator.html')

@app.route('/text_to_speech')
def text_to_speech_page():
    logging.info("Serving text_to_speech.html")
    return render_template('text_to_speech.html')

# --- API Endpoints ---

@app.route('/api/summarize', methods=['POST'])
def summarize_api():
    logging.info("Received /api/summarize POST request.")
    data = request.get_json()
    text = data.get('text')
    length = float(data.get('length', 50)) / 100 # Convert percentage to ratio

    if not text:
        logging.warning("No text provided for summarization.")
        return jsonify({"summary": "", "error": "Please provide text to summarize."}), 400

    try:
        summary = summarize_text(text, length)
        if "Error: Summarizer model not loaded." in summary:
            logging.error("Summarizer model not loaded for API request.")
            return jsonify({"summary": "", "error": summary}), 500
        logging.info("Text summarization successful.")
        return jsonify({"summary": summary.strip()})
    except Exception as e:
        humanized_text = f"An unexpected error occurred during summarization: {str(e)}"
        logging.error(humanized_text, exc_info=True)
        return jsonify({"summary": "", "error": humanized_text}), 500


@app.route('/api/rewrite', methods=['POST'])
def rewrite_api():
    logging.info("Received /api/rewrite POST request.")
    data = request.get_json()
    text = data.get('text')
    creativity_ratio = data.get('creativity', 0.5) # Default to 0.5 if not provided

    if not text:
        logging.warning("No text provided for rewriting.")
        return jsonify({"rewritten_text": "", "error": "Please provide text to rewrite."}), 400

    try:
        rewritten_text = rewrite_article(text, creativity_ratio)
        if "Error: Paraphraser model not loaded." in rewritten_text:
            logging.error("Paraphraser model not loaded for API request.")
            return jsonify({"rewritten_text": "", "error": rewritten_text}), 500
        logging.info("Article rewriting successful.")
        # MODIFICATION: Return a JSON object with 'rewritten_text' key
        return jsonify({"rewritten_text": rewritten_text.strip()})
    except Exception as e:
        humanized_text = f"An unexpected error occurred during rewriting: {str(e)}"
        logging.error(humanized_text, exc_info=True)
        return jsonify({"rewritten_text": "", "error": humanized_text}), 500


@app.route('/api/humanize', methods=['POST'])
def humanize_api():
    logging.info("Received /api/humanize POST request.")
    data = request.get_json()
    text = data.get('text')
    creativity = data.get('creativity', 0.7) # Default creativity level

    if not text:
        logging.warning("No text provided for humanization.")
        return jsonify({"humanized_text": "", "error": "Please provide text to humanize."}), 400

    try:
        humanized_text = humanize_text_content(text, creativity)
        # Check for error from Gemini function
        if isinstance(humanized_text, str) and humanized_text.startswith("Error:"):
            logging.error(f"Humanization API call failed: {humanized_text}")
            return jsonify({"humanized_text": "", "error": humanized_text}), 500
        
        logging.info("Text humanization successful.")
        return jsonify({"humanized_text": humanized_text.strip()})
    except Exception as e:
        humanized_text = f"An unexpected error occurred during humanization: {str(e)}"
        logging.error(humanized_text, exc_info=True)
        return jsonify({"humanized_text": "", "error": humanized_text}), 500


@app.route('/api/generate_email', methods=['POST'])
def generate_email_api():
    logging.info("Received /api/generate_email POST request.")
    data = request.get_json()
    subject = data.get('subject')
    purpose = data.get('purpose')
    recipient = data.get('recipient', '')

    if not subject and not purpose:
        logging.warning("No subject or purpose provided for email generation.")
        return jsonify({"generated_email": "", "error": "Please provide a subject or purpose for the email."}), 400

    try:
        generated_email = generate_email_content(subject, purpose, recipient)
        # Check for error from Gemini function
        if isinstance(generated_email, str) and generated_email.startswith("Error:"):
            logging.error(f"Email generation API call failed: {generated_email}")
            return jsonify({"generated_email": "", "error": generated_email}), 500
        
        logging.info("Email generation successful.")
        return jsonify({"generated_email": generated_email.strip()})
    except Exception as e:
        humanized_text = f"An unexpected error occurred during email generation: {str(e)}"
        logging.error(humanized_text, exc_info=True)
        return jsonify({"generated_email": "", "error": humanized_text}), 500


@app.route('/api/generate_content_ideas', methods=['POST'])
def generate_content_ideas_api():
    logging.info("Received /api/generate_content_ideas POST request.")
    data = request.get_json()
    keywords = data.get('keywords')

    if not keywords:
        logging.warning("No keywords provided for content idea generation.")
        return jsonify({"content_ideas": "", "error": "Please provide keywords for content ideas."}), 400
    
    try:
        content_ideas_text = generate_content_ideas(keywords)
        # Check for error from Gemini function
        if isinstance(content_ideas_text, str) and content_ideas_text.startswith("Error:"):
            logging.error(f"Content ideas API call failed: {content_ideas_text}")
            return jsonify({"content_ideas": "", "error": content_ideas_text}), 500
        
        logging.info("Content idea generation successful.")
        return jsonify({"content_ideas": content_ideas_text.strip()})
    except Exception as e:
        humanized_text = f"An unexpected error occurred during content idea generation: {str(e)}"
        logging.error(humanized_text, exc_info=True)
        return jsonify({"content_ideas": "", "error": humanized_text}), 500


@app.route('/api/generate_slogan', methods=['POST'])
def generate_slogan_api():
    logging.info("Received /api/generate_slogan POST request.")
    data = request.get_json()
    keywords = data.get('keywords')
    num_slogans = data.get('num_slogans', 5) # Safely get num_slogans, default to 5

    if not keywords:
        logging.warning("No keywords provided for slogan generation.")
        return jsonify({"slogans": [], "error": "Please provide keywords for slogan generation."}), 400
    
    # Ensure num_slogans is an int and capped. Added a more robust check.
    try:
        num_slogans = int(num_slogans)
        num_slogans = max(1, min(num_slogans, 10))
    except (ValueError, TypeError):
        logging.warning(f"Invalid num_slogans received: {num_slogans}. Defaulting to 5.")
        num_slogans = 5

    try:
        slogans = generate_slogans(keywords, num_slogans)
        # Check if the slogans contain an error message from the Gemini integration
        if slogans and isinstance(slogans, list) and slogans[0].startswith("Error: Gemini API"):
             logging.error(f"Gemini slogan generation failed: {slogans[0]}")
             return jsonify({"slogans": [], "error": slogans[0]}), 500
        
        logging.info("Slogan generation successful.")
        return jsonify({"slogans": slogans})
    except Exception as e:
        humanized_text = f"An unexpected error occurred during slogan generation: {str(e)}"
        logging.error(humanized_text, exc_info=True)
        return jsonify({"slogans": [], "error": humanized_text}), 500


@app.route('/api/check_plagiarism_ai', methods=['POST'])
def check_plagiarism_ai_api():
    logging.info("Received /api/check_plagiarism_ai POST request.")
    data = request.get_json()
    text = data.get('text')

    if not text:
        logging.warning("No text provided for plagiarism/AI check.")
        return jsonify({"error": "Please provide text to check."}), 400

    try:
        results = check_plagiarism_and_ai(text)
        logging.info("Plagiarism and AI check simulated successfully.")
        return jsonify(results)
    except Exception as e:
        humanized_text = f"An unexpected error occurred during plagiarism/AI check: {str(e)}"
        logging.error(humanized_text, exc_info=True)
        return jsonify({"error": humanized_text}), 500


if __name__ == '__main__':
    # Import 'json' only if needed for Gemini's structured output parsing
    import json 
    
    logging.info("Starting Flask development server. (Main process)")
    # Pre-load only T5 models as other functionalities will use Gemini.
    load_summarizer_model()
    load_paraphraser_model()

    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
