# Model Card: AdaptivePhishingDetector


## Model Details
- **Architecture:** BERT-based email embedding, Frozen BERT classifier, and DQN agent for adaptive RL fine-tuning
- **Baselines:** Random Forest, XGBoost, Frozen BERT
- **Task:** Phishing email detection (binary classification)
- **Input:** Email text or PDF (via CLI or web app)
- **Output:** Phishing/Legitimate prediction with confidence score and probability breakdown


## Intended Use
- Detect phishing emails in enterprise, research, or educational settings
- Demonstrate RL-based model adaptation for adversarial and imbalanced spam
- Benchmark against baselines and frozen BERT for robust evaluation


## Limitations
- Trained on public datasets; may not generalize to all email types
- Requires BERT embeddings; large models may need GPU
- Web app demo is for research, not production
- CLI and web app require proper preprocessing and model loading


## Ethical Considerations
- Do not use for critical security decisions without further validation
- May have biases from training data
- Users should validate on their own data before deployment


## Reproducibility Checklist
- [x] All code and scripts are included
- [x] Datasets and embeddings are documented
- [x] Environment setup instructions provided (see SETUP.md)
- [x] Results and metrics are reproducible via CLI and web app
- [x] Experiment modes (train only, eval only, train+eval) are supported


## Contact
- Author: Mansi Dhanania
- GitHub: [MansiDhanania](https://github.com/MansiDhanania)
- Email: mansi.dhanania@gmail.com

---

## Workflow Reference
- For full setup, CLI commands, and web app usage, see README.md and SETUP.md.
