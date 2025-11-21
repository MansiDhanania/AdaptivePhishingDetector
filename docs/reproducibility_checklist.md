
# Reproducibility Checklist

This checklist ensures that all steps in the AdaptivePhishingDetector project are reproducible by other users and reviewers.


- [x] All code and scripts are included in the repository
- [x] Datasets and embeddings are documented and available
- [x] Environment setup instructions are provided in SETUP.md
- [x] All dependencies are listed in requirements.txt
- [x] CLI and web app usage are documented in README.md
- [x] Results and metrics are saved in results/ for every experiment mode (train only, eval only, train+eval)
- [x] Experiment logs are saved in logs/
- [x] Model card documents architecture, intended use, and limitations
 

## How to Reproduce
1. Clone the repository
2. Set up the environment (see SETUP.md)
3. Prepare data and generate BERT embeddings (see SETUP.md)
4. Run experiments via CLI for baselines, BERT, and DQN+BERT (see README.md)
5. View results and logs in results/ and logs/
6. Launch the web app for interactive demo and prediction
7. Reference model_card.md for architecture and intended use
