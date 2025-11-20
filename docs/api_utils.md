# API Documentation: src/utils

This document summarizes the main public functions and classes in the `src/utils` module. For full details, see code docstrings.

## baseline_models.py
- `train_baseline_models(X, y)`: Train Random Forest and XGBoost on features X and labels y.
- `save_baseline_models(results)`: Save trained baseline models and vectorizer.

## dqn_finetune.py
- `DQNAgent`: Reinforcement learning agent for BERT fine-tuning.
- `train_with_dqn(model, train_loader, val_loader)`: Train BERT using DQN agent.
- `gradual_unfreeze(model, step)`: Gradually unfreeze BERT layers for RL.

## evaluate.py
- `evaluate_accuracy(model, dataloader)`: Compute accuracy for a model.
- `get_predictions(model, dataloader)`: Get predictions and labels.
- `evaluate_model(model, test_loader, model_name, return_probs)`: Full evaluation with metrics and confusion matrix.
- `plot_curves(labels, probs, model_name)`: Plot ROC and PR curves.

## phishing_predictor.py
- `predict_phishing(pdf_path, model, vectorizer)`: Predict phishing from PDF/email file.

## preprocessing.py
- `load_data(csv_path)`: Load and preprocess data from CSV.
- `get_bert_embeddings(texts)`: Generate BERT embeddings for input texts.

## train.py
- `FrozenBERTClassifier`: BERT-based classifier for phishing detection.
- `train_bert_model(embeddings_path)`: Train BERT classifier on embeddings.
- `create_dataloaders(X, y)`: Create PyTorch dataloaders for training/validation/testing.
