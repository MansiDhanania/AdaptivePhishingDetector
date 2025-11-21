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


## 4. Data Preparation & Preprocessing
- Place raw datasets in the `data/` folder.
- Convert CSVs to BERT embeddings for model training:
```powershell
python src/utils/preprocessing.py --input data/phishing_email.csv --output models/phishing_email_bert_embeddings.pt
python src/utils/preprocessing.py --input data/CEAS_08.csv --output models/ceas_bert_embeddings.pt
python src/utils/preprocessing.py --input data/Ling.csv --output models/ling_bert_embeddings.pt
python src/utils/preprocessing.py --input data/SpamAssasin.csv --output models/spamassasin_bert_embeddings.pt
```
Repeat for other datasets as needed.


## 5. Running Experiments

You can run baselines, BERT, or DQN+BERT experiments in three modes: train only, eval only, or train+eval. Results and logs are saved in subfolders for each run.

### Baseline Models (Random Forest, XGBoost)
```powershell
# Train and evaluate
python src/run_experiments.py --mode baselines --train data/phishing_email.csv --eval data/CEAS_08.csv --results results
# Train only
python src/run_experiments.py --mode baselines --train data/phishing_email.csv --train_only --results results
# Eval only (requires previously saved models)
python src/run_experiments.py --mode baselines --eval data/CEAS_08.csv --eval_only --results results
```

### Frozen BERT Classifier
```powershell
# Train and evaluate
python src/run_experiments.py --mode bert --train models/phishing_email_bert_embeddings.pt --eval models/ceas_bert_embeddings.pt --results results
# Train only
python src/run_experiments.py --mode bert --train models/phishing_email_bert_embeddings.pt --train_only --results results
# Eval only (requires previously saved model)
python src/run_experiments.py --mode bert --eval models/ceas_bert_embeddings.pt --eval_only --results results
```

### DQN+BERT (RL Fine-Tuning)
```powershell
# Train and evaluate
python src/run_experiments.py --mode dqn --train models/phishing_email_bert_embeddings.pt --eval models/ceas_bert_embeddings.pt --results results
# Train only
python src/run_experiments.py --mode dqn --train models/phishing_email_bert_embeddings.pt --train_only --results results
# Eval only (requires previously saved model)
python src/run_experiments.py --mode dqn --eval models/ceas_bert_embeddings.pt --eval_only --results results
```

### Results & Logs
- All results (classification reports, plots, model weights) are saved in the specified results subfolder for each run.
- Logs for each experiment are saved in the `logs/` directory.


## 6. Running the Web App
```powershell
python app/app.py
```
Visit `http://localhost:5000` in your browser. Upload a PDF or text email to get predictions and confidence scores.


## 7. Troubleshooting & Notes
- For GPU support, ensure PyTorch is installed with CUDA.
- For advanced users, consider using `environment.yml` for Conda environments.
- If you encounter CUDA/CPU errors, check your PyTorch installation and device availability.
- For dataset errors, verify CSV format and column names (`text_combined`, `label`).
- See README for more details and advanced usage.
