
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
## Directory Structure
```
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
python src/utils/preprocessing.py --input data/CEAS_08.csv --output models/ceas_bert_embeddings.pt
python src/utils/preprocessing.py --input data/Ling.csv --output models/ling_bert_embeddings.pt
python src/utils/preprocessing.py --input data/SpamAssasin.csv --output models/spamassasin_bert_embeddings.pt
```
Repeat for other datasets as needed.

### 3. Training & Evaluation Scenarios

You can run training only, evaluation only, or both (train+eval) for each model type. Results are saved in separate folders for easy comparison:

#### Baseline Models (Random Forest, XGBoost)
Train and evaluate baselines:
```powershell
# Train and evaluate
python src/run_experiments.py --mode baselines --train data/phishing_email.csv --eval data/CEAS_08.csv --results results
# Train only
python src/run_experiments.py --mode baselines --train data/phishing_email.csv --train_only --results results
# Eval only (requires previously saved models)
python src/run_experiments.py --mode baselines --eval data/CEAS_08.csv --eval_only --results results
```
Results saved in `results/train_only/`, `results/eval_only/`, or `results/train_and_eval/`.

#### Frozen BERT Classifier
# Train and evaluate
python src/run_experiments.py --mode bert --train models/phishing_email_bert_embeddings.pt --eval models/ceas_bert_embeddings.pt --results results
# Train only
Results saved in `results/train_only/`, `results/eval_only/`, or `results/train_and_eval/`.

#### DQN+BERT (RL Fine-Tuning)
Train and evaluate DQN+BERT:
```powershell
# Train and evaluate
python src/run_experiments.py --mode dqn --train models/phishing_email_bert_embeddings.pt --train_only --results results
# Eval only (requires previously saved model)
python src/run_experiments.py --mode dqn --eval models/ceas_bert_embeddings.pt --eval_only --results results
```
Results saved in `results/train_only/`, `results/eval_only/`, or `results/train_and_eval/`.

### 4. Results & Logs

- All results (classification reports, plots, model weights) are saved in the specified results subfolder for each run.
- Logs for each experiment are saved in the `logs/` directory.

### 5. Run the Web App

## Directory Structure
```

Start the Flask app for interactive phishing detection:
```powershell
python app/app.py
```
Visit `http://localhost:5000` in your browser. Upload a PDF or text email to get predictions and confidence scores.

---



## Advanced Usage & Custom Experiments

- You can mix and match any train/eval datasets for benchmarking generalization.
- All experiment types (baselines, bert, dqn) support train only, eval only, or train+eval modes.
- Results and logs are always saved in the appropriate subfolders for reproducibility.


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
