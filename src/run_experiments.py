"""
Run experiments for baseline models, BERT, and DQN+BERT.
Allows flexible train/eval dataset selection and generates evaluation plots.
"""
import torch
from src.utils.baseline_models import train_baseline_models, save_baseline_models
from src.utils.train import train_bert_model, FrozenBERTClassifier, create_dataloaders
from src.utils.dqn_finetune import train_with_dqn
from src.utils.evaluate import evaluate_model, plot_curves
from src.utils.preprocessing import load_data
from sklearn.metrics import classification_report, roc_curve, auc, precision_recall_curve
import numpy as np
import argparse
import os
import sys
import logging

# Baseline models: RF and XGBoost

def run_baselines(train_csv, eval_csv, save_dir="results", log_dir="logs"):
    X_train, Y_train = load_data(train_csv)
    X_eval, Y_eval = load_data(eval_csv)
    results = train_baseline_models(X_train, Y_train)
    save_baseline_models(results)
    rf_model, xgb_model, vectorizer = results['rf_model'], results['xgb_model'], results['vectorizer']
    X_eval_vec = vectorizer.transform(X_eval)
    rf_preds = rf_model.predict(X_eval_vec)
    xgb_preds = xgb_model.predict(X_eval_vec)
    rf_probs = rf_model.predict_proba(X_eval_vec)[:, 1]
    xgb_probs = xgb_model.predict_proba(X_eval_vec)[:, 1]
    rf_report = classification_report(Y_eval, rf_preds)
    xgb_report = classification_report(Y_eval, xgb_preds)
    print("Random Forest:\n", rf_report)
    print("XGBoost:\n", xgb_report)
    # Save reports
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "rf_classification_report.txt"), "w") as f:
        f.write(rf_report)
    with open(os.path.join(save_dir, "xgb_classification_report.txt"), "w") as f:
        f.write(xgb_report)
    # Save plots
    import matplotlib.pyplot as plt
    plt.figure()
    plot_curves(Y_eval, rf_probs, "Random Forest")
    plt.savefig(os.path.join(save_dir, "rf_roc_pr.png"))
    plt.close()
    plt.figure()
    plot_curves(Y_eval, xgb_probs, "XGBoost")
    plt.savefig(os.path.join(save_dir, "xgb_roc_pr.png"))
    plt.close()

# BERT classifier

def run_bert(train_pt, eval_pt, save_dir="results"):
    model, _, _, _ = train_bert_model(train_pt)
    eval_data = torch.load(eval_pt)
    X_eval, Y_eval = eval_data['embeddings'], eval_data['labels']
    _, _, test_loader = create_dataloaders(X_eval, Y_eval)
    probs, preds, labels = evaluate_model(model, test_loader, "FrozenBERT", return_probs=True)
    # Save report
    os.makedirs(save_dir, exist_ok=True)
    report = classification_report(labels, preds)
    with open(os.path.join(save_dir, "bert_classification_report.txt"), "w") as f:
        f.write(report)
    # Save plot
    import matplotlib.pyplot as plt
    plt.figure()
    plot_curves(labels, probs, "FrozenBERT")
    plt.savefig(os.path.join(save_dir, "bert_roc_pr.png"))
    plt.close()

# DQN+BERT

def run_dqn(train_pt, eval_pt, save_dir="results"):
    data = torch.load(train_pt)
    X, Y = data['embeddings'], data['labels']
    train_loader, val_loader, test_loader = create_dataloaders(X, Y)
    model = FrozenBERTClassifier().to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    model, _, _ = train_with_dqn(model, train_loader, val_loader)
    eval_data = torch.load(eval_pt)
    X_eval, Y_eval = eval_data['embeddings'], eval_data['labels']
    _, _, test_loader_eval = create_dataloaders(X_eval, Y_eval)
    probs, preds, labels = evaluate_model(model, test_loader_eval, "DQN+BERT", return_probs=True)
    # Save report
    os.makedirs(save_dir, exist_ok=True)
    report = classification_report(labels, preds)
    with open(os.path.join(save_dir, "dqn_bert_classification_report.txt"), "w") as f:
        f.write(report)
    # Save plot
    import matplotlib.pyplot as plt
    plt.figure()
    plot_curves(labels, probs, "DQN+BERT")
    plt.savefig(os.path.join(save_dir, "dqn_bert_roc_pr.png"))
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Run phishing detection experiments.")
    parser.add_argument("--mode", choices=["baselines", "bert", "dqn"], required=True, help="Experiment type to run.")
    parser.add_argument("--train", required=True, help="Training dataset (csv or pt)")
    parser.add_argument("--eval", required=True, help="Evaluation dataset (csv or pt)")
    parser.add_argument("--results", default="results", help="Directory to save results")
    parser.add_argument("--logs", default="logs", help="Directory to save logs")
    args = parser.parse_args()
    # Setup logging
    os.makedirs(args.logs, exist_ok=True)
    logging.basicConfig(filename=os.path.join(args.logs, f"{args.mode}_experiment.log"), level=logging.INFO)
    logging.info(f"Running {args.mode} experiment: train={args.train}, eval={args.eval}")
    if args.mode == "baselines":
        run_baselines(args.train, args.eval, save_dir=args.results, log_dir=args.logs)
    elif args.mode == "bert":
        run_bert(args.train, args.eval, save_dir=args.results)
    elif args.mode == "dqn":
        run_dqn(args.train, args.eval, save_dir=args.results)
    else:
        print("Invalid mode.")
        sys.exit(1)

if __name__ == "__main__":
    main()
