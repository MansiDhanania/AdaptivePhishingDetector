
# AdaptivePhishingDetector

AdaptivePhishingDetector is a modular pipeline for phishing email detection using reinforcement learning (DQN) to fine-tune BERT. The system is designed for robust generalization to adversarial and imbalanced email datasets, and includes a web app for real-world deployment.

## Key Features
- Modular RL pipeline for BERT fine-tuning (DQN+BERT)
- Baseline models (Random Forest, XGBoost) for benchmarking
- Flexible data preprocessing and experiment configuration
- CLI-driven workflow for reproducibility
- Web app for interactive phishing detection (PDF/text input)
- GPU/CPU compatibility throughout

## Directory Structure
```
environment.yml         # Conda environment (recommended)
README.md              # Project documentation
SETUP.md               # Setup instructions
app/                   # Flask web app
  app.py
  index.html
  requirements.txt
data/                  # Raw CSV datasets
  CEAS_08.csv
  Ling.csv
  phishing_email.csv
  SpamAssasin.csv
docs/                  # Documentation
models/                # Saved models and BERT embeddings (.pt)
results/               # Experiment results, metrics, plots
src/                   # Core logic and experiment runner
  run_experiments.py
  utils/
	baseline_models.py
	dqn_finetune.py
	evaluate.py
	phishing_predictor.py
	preprocessing.py
	train.py
```


## Quickstart: End-to-End Workflow

### 1. Environment Setup

**Recommended:**
```powershell
conda env create -f environment.yml
conda activate <your_env_name>
```
Or (if using pip):
```powershell
pip install -r app/requirements.txt
```

### 2. Data Preprocessing: CSV → BERT Embeddings

Convert raw CSVs to BERT `.pt` embeddings for model training:
```powershell
python src/utils/preprocessing.py --input data/phishing_email.csv --output models/phishing_email_bert_embeddings.pt
```
Repeat for other datasets as needed.

### 3. Training & Evaluation Scenarios

You can now run training, evaluation, or both, and results will be saved in separate folders for easy comparison:

#### Train Only
```powershell
python src/run_experiments.py --mode dqn --train models/phishing_email_bert_embeddings.pt --train_only --results results
```
Results saved in `results/train_only/`

#### Eval Only (with a previously trained model)
```powershell
python src/run_experiments.py --mode dqn --eval models/phishing_email_bert_embeddings.pt --eval_only --results results
```
Results saved in `results/eval_only/`

#### Train and Eval (default)
```powershell
python src/run_experiments.py --mode dqn --train models/phishing_email_bert_embeddings.pt --eval models/phishing_email_bert_embeddings.pt --results results
```
Results saved in `results/train_and_eval/`

### 4. Run the Web App

Start the Flask app for interactive phishing detection:
```powershell
python app/app.py
```
Visit `http://localhost:5000` in your browser. Upload a PDF or text email to get predictions and confidence scores.

---


## Advanced Usage: Baselines, Evaluation & Custom Experiments

Run baseline models, custom splits, or evaluate a previously trained model:
```powershell
# Baselines (Random Forest, XGBoost)
python src/run_experiments.py --mode baselines --train data/phishing_email.csv --eval data/CEAS_08.csv

# Frozen BERT
python src/run_experiments.py --mode bert --train models/phishing_email_bert_embeddings.pt --eval models/CEAS_bert_embeddings.pt

# Evaluate a previously trained DQN+BERT model
# (Use the same .pt file for --train if you only want to load weights and skip retraining)
python src/run_experiments.py --mode dqn --train models/phishing_email_bert_embeddings.pt --eval models/CEAS_bert_embeddings.pt --results results
```
Results and logs are saved in `results/` and `logs/`.

**Note:**
- The script will load the saved model weights and run evaluation on the specified dataset.
- If you only want to evaluate (not retrain), ensure the script logic supports loading the model and skipping training. If not, add a flag or modify the script to only run evaluation.

## Technical Notes
- All scripts support GPU/CPU automatically
- Modular codebase for easy extension and reproducibility
- Results, metrics, and plots are saved for every run
- See `docs/` for model card, API details, and reproducibility checklist

## Troubleshooting
- Ensure all dependencies are installed (see above)
- If you encounter CUDA/CPU errors, check your PyTorch installation and device availability
- For dataset errors, verify CSV format and column names (`text_combined`, `label`)

## Citation & Contact
For academic or industrial use, please cite this repository or contact the maintainer via GitHub.
