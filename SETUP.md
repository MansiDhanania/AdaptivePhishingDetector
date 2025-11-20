# Setup Instructions

This guide will help you set up the environment for AdaptivePhishingDetector.

## 1. Clone the Repository
```bash
git clone https://github.com/MansiDhanania/AdaptivePhishingDetector.git
cd AdaptivePhishingDetector
```

## 2. Create and Activate a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

## 3. Install Dependencies
```bash
pip install -r app/requirements.txt
```

## 4. Data Preparation
- Place raw datasets in the `data/` folder.
- Precomputed BERT embeddings should be in the `models/` folder.

## 5. Running Experiments
See the README for CLI usage and experiment instructions.

## 6. Running the Web App
```bash
python app/app.py
```

## 7. Additional Notes
- For GPU support, ensure PyTorch is installed with CUDA.
- For advanced users, consider using `environment.yml` for Conda environments.
