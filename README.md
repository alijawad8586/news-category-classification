# News Category Classification Project

End-to-end Python project for scraping, validating, preprocessing, exploring,
feature-engineering, labeling, and classifying news articles.

## Classification models

The training pipeline compares and tunes four requested model families:

- Logistic Regression
- Decision Tree
- Random Forest
- Linear SVM

`label/train_all_models.py` removes duplicate text before splitting, uses a
stratified train/test split, creates sparse word/bigram TF-IDF features, and
trains 12 hyperparameter candidates with up to 12 parallel workers. It saves
the best model from each family, a best-overall model, accuracy summaries,
classification reports, and confusion matrices locally.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
playwright install chromium
```

## Train the classifiers

Place `balanced_news_data.csv` inside the `label` directory. It must contain a
`category` column plus suitable title and content columns.

```powershell
cd label
python train_all_models.py --output-dir model_outputs_4models
```

The dataset and generated outputs are intentionally excluded from GitHub.

## Database configuration

Database scripts read the PostgreSQL password from the environment:

```powershell
$env:DB_PASSWORD = "your-password"
```

Do not commit passwords, connection secrets, or raw datasets.
