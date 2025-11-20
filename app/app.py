from flask import Flask, request, render_template, jsonify
import sys
sys.path.append('..')

from src.utils.phishing_predictor import predict_phishing, load_trained_model, extract_text_from_pdf
from src.utils.train import FrozenBERTClassifier
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load model on startup
MODEL_PATH = '../models/dqn_finetuned_bert.pth'
model = load_trained_model(MODEL_PATH, FrozenBERTClassifier)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        
        if file.filename == '' or file.filename is None:
            return jsonify({'error': 'No file selected'}), 400
        
        if file and file.filename.endswith('.pdf'):
            # Save uploaded file - now filename is guaranteed to be a string
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            
            # Make prediction
            result = predict_phishing(filepath, model)
            
            # Clean up
            os.remove(filepath)
            
            return jsonify(result)
        else:
            return jsonify({'error': 'Only PDF files are supported'}), 400
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict_text', methods=['POST'])
def predict_text():
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'Invalid JSON data'}), 400
        
        text = data.get('text', '')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        result = predict_phishing(text, model)
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)