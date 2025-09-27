import pandas as pd
import nltk
from models.stacking_pipeline import train_ultimate_ensemble_model

def ensure_nltk():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)

def main():
    ensure_nltk()
    train_df = pd.read_csv("train.csv")
    val_df = pd.read_csv("validation.csv")
    test_df = pd.read_csv("test.csv")

    submission_df, model, scaler, selector, threshold = train_ultimate_ensemble_model(
        train_df, val_df, test_df, text_column='text', label_column='label'
    )
    print("\nFirst few predictions:")
    print(submission_df.head(10))

if __name__ == "__main__":
    main()
