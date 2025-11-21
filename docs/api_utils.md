
# API Documentation: src/utils

This document summarizes the main public functions and classes in the `src/utils` module. For full details, see code docstrings and the README for CLI usage and workflow.


## baseline_models.py
- `train_baseline_models(X, y)`: Train Random Forest and XGBoost on features X and labels y.
- `save_baseline_models(results)`: Save trained baseline models and vectorizer.
- `load_baseline_models(save_dir)`: Load previously saved baseline models and vectorizer.


## dqn_finetune.py
- `DQNAgent`: Reinforcement learning agent for BERT fine-tuning.
- `train_with_dqn(model, train_loader, val_loader, epochs)`: Train BERT using DQN agent for specified epochs.
- `gradual_unfreeze(model, epoch, epochs, state)`: Gradually unfreeze BERT layers for RL.


## evaluate.py
- `evaluate_accuracy(model, dataloader)`: Compute accuracy for a model.
- `get_predictions(model, dataloader)`: Get predictions and labels.
- `evaluate_model(model, test_loader, model_name, return_probs)`: Full evaluation with metrics and confusion matrix.
- `plot_curves(labels, probs, model_name)`: Plot ROC and PR curves. Plots are saved in results folders for each experiment.


## phishing_predictor.py
- `predict_phishing(pdf_path_or_text, model)`: Predict phishing from PDF or text input using loaded model. Returns prediction, confidence, and probability breakdown.


## preprocessing.py
- `load_data(csv_path)`: Load and preprocess data from CSV. Expects columns `text_combined` and `label`.
- `get_bert_embeddings(texts)`: Generate BERT embeddings for input texts. Used for preprocessing before model training.


## train.py
- `FrozenBERTClassifier`: BERT-based classifier for phishing detection.
- `train_bert_model(embeddings_path, epochs, learning_rate, save_path)`: Train BERT classifier on embeddings and save weights.
- `create_dataloaders(X, y, device)`: Create PyTorch dataloaders for training/validation/testing.

---

## CLI & Web App Integration
- All core functions are used via `src/run_experiments.py` for CLI workflows (see README for commands).
- The web app (`app/app.py`) loads trained models and uses `phishing_predictor.py` for real-time predictions from PDF/text uploads.
