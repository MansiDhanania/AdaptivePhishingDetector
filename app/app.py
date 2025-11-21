
import os
import sys
import logging
from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from utils.phishing_predictor import predict_phishing, load_trained_model, extract_text_from_pdf
from utils.train import FrozenBERTClassifier

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size


# Setup logging
logging.basicConfig(filename='logs/webapp.log', level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


# Load model on startup using a path relative to app.py
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'dqn_finetuned_bert.pth')
model = load_trained_model(MODEL_PATH, FrozenBERTClassifier)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            logging.error('No file uploaded')
            return jsonify({'error': 'No file uploaded'}), 400
        file = request.files['file']
        if file.filename == '' or file.filename is None:
            logging.error('No file selected')
            return jsonify({'error': 'No file selected'}), 400
        if file and file.filename.endswith('.pdf'):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            result = None
            try:
                result = predict_phishing(filepath, model)
            except Exception as pred_err:
                logging.error(f'Prediction error: {pred_err}')
                if os.path.exists(filepath):
                    os.remove(filepath)
                return jsonify({'error': f'Prediction error: {pred_err}'}), 500
            finally:
                if os.path.exists(filepath):
                    os.remove(filepath)
            if not result or (isinstance(result, dict) and result.get('error')):
                logging.error(f'Prediction failed: {result}')
                return jsonify({'error': result.get('error', 'Unknown error during prediction')}), 500
            logging.info(f'Prediction successful for file: {file.filename} | Result: {result}')
            return jsonify(result)
        else:
            logging.error('Only PDF files are supported')
            return jsonify({'error': 'Only PDF files are supported'}), 400
    except Exception as e:
        logging.error(f'Unexpected error: {e}')
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500

@app.route('/predict_text', methods=['POST'])
def predict_text():
    try:
        data = request.get_json()
        if not data:
            logging.error('Invalid JSON data')
            return jsonify({'error': 'Invalid JSON data'}), 400
        text = data.get('text', '')
        if not text:
            logging.error('No text provided')
            return jsonify({'error': 'No text provided'}), 400
        try:
            result = predict_phishing(text, model)
        except Exception as pred_err:
            logging.error(f'Prediction error: {pred_err}')
            return jsonify({'error': f'Prediction error: {pred_err}'}), 500
        logging.info(f'Prediction successful for text input | Result: {result}')
        return jsonify(result)
    except Exception as e:
        logging.error(f'Unexpected error: {e}')
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)