# Model Card: AdaptivePhishingDetector

## Model Details
- **Architecture:** BERT-based email embedding + DQN agent for adaptive fine-tuning
- **Baselines:** Random Forest, XGBoost, Frozen BERT
- **Task:** Phishing email detection (binary classification)
- **Input:** Email text or PDF
- **Output:** Spam/Phishing prediction with confidence score

## Intended Use
- Detect phishing emails in enterprise or research settings
- Demonstrate RL-based model adaptation for adversarial spam

## Limitations
- Trained on public datasets; may not generalize to all email types
- Requires BERT embeddings; large models may need GPU
- Web app demo is for research, not production

## Ethical Considerations
- Do not use for critical security decisions without further validation
- May have biases from training data

## Reproducibility Checklist
- [x] All code and scripts are included
- [x] Datasets and embeddings are documented
- [x] Environment setup instructions provided
- [x] Results and metrics are reproducible via CLI

## Contact
- Author: Mansi Dhanania
- GitHub: [MansiDhanania](https://github.com/MansiDhanania)
- Email: mansi.dhanania@gmail.com
