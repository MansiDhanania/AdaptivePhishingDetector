
# AdaptivePhishingDetector
A reinforcement learning-enhanced phishing email detection system, designed to address the evolving challenges of adversarial spam and class imbalance in real-world email environments.

## Project Overview
This repository demonstrates that a DQN-based reinforcement learning program for fine-tuning BERT (a small LLM) improves phishing email detection compared to:
- Baseline models (Random Forest, XGBoost)
- Simple BERT training on a dataset

The DQN+BERT approach is designed to generalize better to new/unseen phishing instances, even if not trained on those specific data distributions.

## Datasets
- `phishing_email.csv`: Combines 6 popular phishing datasets (including CEAS, Ling, SpamAssassin)
- `CEAS_08.csv`, `Ling.csv`, `SpamAssasin.csv`: Individual datasets for cross-dataset evaluation

## Modular Experiment Pipeline
All experiments are run via `src/run_experiments.py`. You can flexibly specify which dataset to use for training and which for evaluation.

### How to Run Experiments
1. **Baselines (RF, XGBoost):**
	- Trains on one dataset, evaluates on another
2. **Frozen BERT Classifier:**
	- Trains on BERT embeddings, evaluates on another dataset
3. **DQN+BERT Fine-tuning:**
	- Trains with RL agent, evaluates on another dataset


#### CLI Usage
Run any experiment from the command line:
```bash
# Baselines (Random Forest, XGBoost)
python src/run_experiments.py --mode baselines --train data/phishing_email.csv --eval data/CEAS_08.csv

# Frozen BERT
python src/run_experiments.py --mode bert --train models/phishing_email_bert_embeddings.pt --eval models/CEAS_bert_embeddings.pt

# DQN+BERT
python src/run_experiments.py --mode dqn --train models/phishing_email_bert_embeddings.pt --eval models/CEAS_bert_embeddings.pt
```
Results and logs are saved in the `results/` and `logs/` folders.

You can modify the script to use any dataset split.


### Evaluation & Plots
- Classification report, confusion matrix, Matthews correlation coefficient
- ROC and Precision-Recall curves for each model (saved in `results/`)
- Experiment logs saved in `logs/`


## Interactive Demo (Web App)
The Flask web app allows you to upload PDF/email files and get phishing predictions with confidence scores.

### How to Run the Web App
1. Install dependencies:
	```bash
	pip install -r app/requirements.txt
	```
2. Start the app:
	```bash
	python app/app.py
	```
3. Open your browser and go to `http://localhost:5000`.
4. Upload a PDF/email and view the prediction and confidence score.

## Deployment
All code is modular. You can remove the original notebook (`ECSE555_FinalProjectCode_AllData.ipynb`) once satisfied.


## Directory Structure
- `src/`: Core logic and experiment runner
- `src/utils/`: Model, training, RL, evaluation, and preprocessing modules
	- `data/`: Raw and processed datasets
	- `models/`: Saved models and BERT embeddings
- `results/`: Experiment results, metrics, and figures
- `logs/`: Experiment logs
- `docs/`: Documentation
- `app/`: Web demo (Flask app)

## Reproducibility
You can easily swap datasets for training/evaluation to demonstrate generalization and robustness.
