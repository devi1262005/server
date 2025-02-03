from flask import Flask, request, jsonify
import os
from werkzeug.utils import secure_filename
from PyPDF2 import PdfReader
from docx import Document
from huggingface_hub import InferenceClient
import re
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Directory to temporarily store uploaded files
UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allowed file extensions
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Initialize Hugging Face Inference Client
client = InferenceClient(api_key=os.getenv("HF_API_KEY"))

@app.route('/get_tags', methods=['POST'])
def get_tags():
    if 'file' in request.files:
        file = request.files['file']

        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            try:
                extracted_keywords = process_resume(file_path)
                return jsonify({"keywords": extracted_keywords}), 200
            except Exception as e:
                return jsonify({"error": str(e)}), 500
            finally:
                os.remove(file_path)

        return jsonify({"error": "Invalid file type"}), 400

    elif 'text' in request.json:
        user_text = request.json.get('text')
        if not user_text:
            return jsonify({"error": "No text provided"}), 400

        try:
            extracted_keywords = process_text(user_text)
            return jsonify({"keywords": extracted_keywords}), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return jsonify({"error": "No valid input provided"}), 400


def call_ai_model(prompt):
    try:
        response = client.text_generation(prompt=prompt, max_new_tokens=100)
        return response.strip().split(',')
    except Exception as e:
        raise ValueError(f"Error during AI completion: {str(e)}")


def process_resume(file_path):
    extracted_text = extract_text_from_file(file_path)
    return process_text(extracted_text)


def process_text(text):
    prompt = (
        "You are an advanced AI system specialized in extracting only relevant, strictly technical keywords from resumes. "
        "Extract precisely 100 keywords from the given text, strictly limited to technical skills, programming languages, "
        "frameworks, libraries, cloud services, tools, methodologies, and industry-specific terms. "
        "Do not include greetings, numbers, personal information, job titles, soft skills, or general words like 'development,' "
        "'technology,' 'team,' or 'project.' Avoid any introductory phrases or summaries. "
        "Respond only with a comma-separated list of the extracted technical keywords without additional words or explanations. "
        "The extracted keywords must strictly appear in the given text and should not be guessed or inferred.\n\n"
        f"Text:\n{text}"
    )

    keywords = call_ai_model(prompt)  # Get response from AI
    
    # Clean up keywords
    cleaned_keywords = [
        word.strip(",").strip()
        for word in keywords
        if word.strip()
    ]
    
    # Ensure 100 unique keywords
    unique_keywords = list(set(cleaned_keywords))  # Remove duplicates
    if len(unique_keywords) > 100:
        return unique_keywords[:100]
    elif len(unique_keywords) < 100:
        additional_keywords = get_non_significant_keywords(text, 100 - len(unique_keywords))
        return unique_keywords + additional_keywords
    return unique_keywords


def get_non_significant_keywords(text, num_keywords):
    """
    Extracts additional words from the text if there are not enough technical keywords.
    """
    words = [word.strip(",.") for word in text.split() if len(word.strip(",.")) > 2]
    return words[:num_keywords]


def extract_text_from_pdf(file_path):
    try:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted
        return text
    except Exception as e:
        raise ValueError(f"Error extracting text from PDF: {str(e)}")


def extract_text_from_docx(file_path):
    try:
        document = Document(file_path)
        text = []
        for paragraph in document.paragraphs:
            text.append(paragraph.text)
        return "\n".join(text)
    except Exception as e:
        raise ValueError(f"Error extracting text from DOCX: {str(e)}")


def extract_text_from_txt(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        raise ValueError(f"Error extracting text from TXT: {str(e)}")


def extract_text_from_file(file_path):
    """
    Determines the file type and extracts text accordingly.
    """
    if file_path.endswith('.pdf'):
        return extract_text_from_pdf(file_path)
    elif file_path.endswith('.docx'):
        return extract_text_from_docx(file_path)
    elif file_path.endswith('.txt'):
        return extract_text_from_txt(file_path)
    else:
        raise ValueError("Unsupported file type")


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
