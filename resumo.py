import os
import logging
import re
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from PyPDF2 import PdfReader
from docx import Document
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Logging setup
logging.basicConfig(level=logging.INFO)

# Uploads directory
UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allowed file types
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt'}

# Hugging Face API Client
client = InferenceClient(api_key=os.getenv("HF_API_KEY"))

# Unwanted non-technical words
UNWANTED_TEXT = {
    "here", "are", "the", "keywords", "from", "text", "separated",
    "by", "commas", "without", "numbers", "greetings", "or", "additional",
    "and", "only", "reject", "unnecessary", "words", "strictly", "technical",
    "tag", "extracted", "code", "50", "text", "<placeholder>", "not", "like"
}

# List of technical keywords
TECHNICAL_KEYWORDS = {
    'flask', 'dart', 'aws', 'artificial intelligence', 'webrtc', 'firestore', 'flutter', 'mongodb',
    'sqlite', 'mysql', 'php', 'firebase', 'c++', 'python', 'nosql', 'api', 'stun', 'turn', 'agenda',
    'real-time', 'websockets', 'video', 'calls', 'full-stack', 'mongo', 'sql', 'html', 'css', 'java',
    'kotlin', 'docker', 'javascript', 'nodejs', 'cloud', 'react', 'angular', 'vue', 'typescript', 'graphql'
}

def allowed_file(filename):
    """Check if the file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_file(file_path):
    """Extract text based on file type."""
    try:
        if file_path.endswith('.pdf'):
            return extract_text_from_pdf(file_path)
        elif file_path.endswith('.docx'):
            return extract_text_from_docx(file_path)
        elif file_path.endswith('.txt'):
            return extract_text_from_txt(file_path)
        else:
            raise ValueError("Unsupported file type")
    except Exception as e:
        logging.error(f"Error extracting text: {e}")
        return ""

def extract_text_from_pdf(file_path):
    """Extract text from a PDF file."""
    reader = PdfReader(file_path)
    return "\n".join(page.extract_text() for page in reader.pages if page.extract_text())

def extract_text_from_docx(file_path):
    """Extract text from a DOCX file."""
    document = Document(file_path)
    return "\n".join(paragraph.text for paragraph in document.paragraphs)

def extract_text_from_txt(file_path):
    """Extract text from a TXT file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def call_ai_model(prompt):
    """Calls the AI model and retrieves keywords."""
    try:
        response = client.text_generation(prompt=prompt, max_new_tokens=100)
        keywords = response.strip().split(',')
        return clean_keywords(keywords)
    except Exception as e:
        logging.error(f"AI Model Error: {e}")
        return []

def clean_keywords(keywords):
    """Filter out unwanted and non-technical keywords."""
    cleaned_keywords = [
        word.strip().lower() for word in keywords
        if word.strip() and word.lower() not in UNWANTED_TEXT and word.isalpha()
    ]
    return list(set(cleaned_keywords))  # Remove duplicates

@app.route('/get_tags', methods=['POST'])
def get_tags():
    """Extract keywords from uploaded file or text input."""
    try:
        if 'file' in request.files:
            file = request.files['file']

            if file.filename == '':
                return jsonify({"error": "No file selected"}), 400

            if not allowed_file(file.filename):
                return jsonify({"error": "Invalid file type"}), 400

            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            text = extract_text_from_file(file_path)
            os.remove(file_path)  # Clean up

        elif 'text' in request.json:
            text = request.json.get('text', '').strip()
            if not text:
                return jsonify({"error": "No text provided"}), 400
        else:
            return jsonify({"error": "No valid input provided"}), 400

        if not text:
            return jsonify({"error": "Empty text extracted"}), 400

        prompt = (
            f"Extract technical keywords from the following text. "
            f"Ensure keywords are related to programming, AI, or software development. "
            f"Do not include greetings, numbers, or irrelevant terms. Provide only a list of keywords, separated by commas:\n\n{text}"
        )

        keywords = call_ai_model(prompt)

        return jsonify({"keywords": keywords}), 200

    except Exception as e:
        logging.error(f"Error in get_tags: {e}")
        return jsonify({"error": "Internal server error", "details": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
