
from flask import Flask, request, jsonify
import os  
from waitress import serve
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

# Initialize the Hugging Face Inference Client using the API key from the environment
client = InferenceClient(api_key=os.getenv("HF_API_KEY"))  # Using the value from .env file

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
            prompt = f"Extract all keywords from the Skills, Employment, and Education sections: {user_text}."
            keywords = call_ai_model(prompt)
            return jsonify({"keywords": keywords}), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return jsonify({"error": "No valid input provided"}), 400

# Unwanted text array
UNWANTED_TEXT = [
    "here", "are", "the", "keywords", "from", "text", "separated", 
    "by", "commas", "without", "numbers", "greetings", "or", 
    "additional", "text", "and", "only", "reject", "uneccessary", "words", "Here are the 100 keywords extracted from the text:" 
    "strictly technical and devoid of non-technical words or phrases","Here are the extracted keywords",
    "not", "like", "technical", "tag", "extracted","commas:", ":", "code", "50","text:", "<placeholder>"
]
common_non_technical_words = {
    'and', 'about', 'here', 'currently', 'new', 'exploring', 'projects', 'diving', 'into', 
    'development', 'calls', 'website', 'mobile', 'app', 'programming', 'tech', 
    'meeting', 'language', 'backend', 'frontend', 'team', 'members', 'hackathon', 'freelancing',
    'contact', 'email', 'number', 'address', 'location', 'number', 
}

# List of strictly technical terms to prioritize (could be expanded as needed)
technical_keywords = {
    'flask', 'dart', 'aws', 'artificial intelligence', 'webrtc', 'firestore', 'flutter', 'mongodb', 
    'sqlite', 'mysql', 'php', 'firebase', 'c++', 'python', 'nosql', 'api', 'stun', 'turn', 'agenda',
    'real-time', 'websockets', 'video', 'calls', 'full-stack', 'mongo', 'sql', 'html', 'css', 'java', 'kotlin',
    'docker', 'javascript', 'nodejs', 'cloud', 'python', 'react', 'angular', 'vue', 'typescript', 'graphql'
}

def process_resume(file_path):
    extracted_text = extract_text_from_file(file_path)
    prompt = (
        f"Extract one hundred keywords from the following text. Keywords must only come from the text provided. "
        f"Avoid greetings, numbers, or additional text. Strictly reject non-technical words or phrases in {UNWANTED_TEXT} and anything similar to {UNWANTED_TEXT} please do not use variants to evade it. Please do not consider greetings in keywords."
        f"Provide keywords separated by commas: {extracted_text}."
        f"Please do not greet the user, or providing starting prompts, just be to the point, no greeting, no interaction."
    )
    keywords = call_ai_model(prompt)  # Assuming this returns a list
    
    # Clean up keywords
    cleaned_keywords = [
        word.strip(",").strip()  # Remove trailing commas and extra whitespace
        for word in keywords
        if word.strip() and word.lower() not in UNWANTED_TEXT
    ]
    
    # Ensure 100 unique keywords
    unique_keywords = list(set(cleaned_keywords))  # Remove duplicates
    if len(unique_keywords) > 100:
        return unique_keywords[:100]
    elif len(unique_keywords) < 100:
        # Find the least significant words in the extracted text to pad the result
        additional_keywords = find_least_significant_keywords(extracted_text, 100 - len(unique_keywords))
        return unique_keywords + additional_keywords
    return unique_keywords

def clean_keywords(keywords):
    """
    Filters out non-technical and irrelevant terms and retains only strictly technical keywords.
    """
    cleaned_keywords = []

    for keyword in keywords:
        # Clean keyword by stripping spaces and converting to lowercase
        keyword = keyword.strip().lower()

        # Ignore common non-technical words
        if keyword in common_non_technical_words:
            continue

        # Only consider keywords that are in the list of technical keywords
        if keyword in technical_keywords:
            cleaned_keywords.append(keyword)

        # Ignore irrelevant patterns (like emails, locations, etc.)
        elif re.search(r'\b\w+@\w+\.\w+', keyword):  # if it's an email address
            continue
        elif re.search(r'\d{3,}', keyword):  # if it's a number or numeric phrase
            continue
        elif len(keyword) > 2 and keyword.isalpha():  # check that keyword is at least 3 characters and contains only letters
            # Add only technical or potentially technical words
            cleaned_keywords.append(keyword)

    # Return unique cleaned keywords
    return list(set(cleaned_keywords))

def find_least_significant_keywords(text, num_keywords):
    """
    Extracts a specified number of technical keywords from the provided text.
    """
    prompt = (
        f"Extract {num_keywords} strictly technical keywords from the following text. "
        f"Do not include common, non-technical, or irrelevant words or phrases. "
        f"Only include technical terms related to programming, tools, technologies, frameworks, or coding languages. "
        f"Provide only keywords, separated by commas:\n\n{text}."
    )

    # Get the AI model's response (assuming the AI API client is available)
    try:
        completion = client.chat.completions.create(
            model="meta-llama/Meta-Llama-3-8B-Instruct",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100
        )
        # Extract the keywords from the AI response
        keywords = completion['choices'][0]['message']['content'].strip().split(',')

        # Clean and filter the keywords
        cleaned_keywords = clean_keywords(keywords)

        # If the number of unique keywords exceeds the desired count, limit the list
        if len(cleaned_keywords) > num_keywords:
            return cleaned_keywords[:num_keywords]
        elif len(cleaned_keywords) < num_keywords:
            # Return additional non-significant terms (such as common terms)
            additional_keywords = get_non_significant_keywords(text, num_keywords - len(cleaned_keywords))
            return cleaned_keywords + additional_keywords

        return cleaned_keywords

    except Exception as e:
        raise ValueError(f"Error during AI completion: {str(e)}")

def get_non_significant_keywords(text, num_keywords):
    # Use a basic AI model or fallback logic to get common, non-significant terms
    words = [word.strip(",.") for word in text.split() if len(word.strip(",.")) > 2]
    return words[:num_keywords]

def extract_text_from_pdf(file_path):
    try:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
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
        with open(file_path, 'r') as file:
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
    serve(app, host="0.0.0.0", port=5000)
