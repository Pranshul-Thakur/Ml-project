# YouTube Sentiment Analyzer

Analyze public sentiment from YouTube video comments using machine learning. Extracts comments via YouTube Data API v3 and classifies sentiment using TF-IDF vectorization and Logistic Regression.

## Features

- Fetch comments from any YouTube video
- Text preprocessing with lemmatization and stopword removal
- Sentiment classification (positive/negative)
- TF-IDF feature extraction
- Custom comment sentiment prediction

## Tech Stack

- YouTube Data API v3
- TextBlob for sentiment labeling
- NLTK for text preprocessing
- Scikit-learn for ML model
- BeautifulSoup for HTML cleaning

## Installation

```bash
pip install -r requirements.txt
```

Download NLTK data:
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

## Setup

1. Get a YouTube Data API key from [Google Cloud Console](https://console.cloud.google.com/)
2. Replace `API_KEY` in `main.py` with your key

## Usage

```python
# Set video ID (from YouTube URL)
video_id = "DOgnE6JQqso"

# Fetch and analyze comments
df = vid_Comment(video_id)
df["Cleaned_Comments"] = df["Comment"].apply(preprocess)
df["Sentiment"] = df["Cleaned_Comments"].apply(get_sentiment)

# Train model
model.fit(x_train_tif, y_train)

# Predict new comments
predict_sentiment("This video is amazing!")
```

## Model Performance

The model uses:
- TF-IDF vectorization with 333 features
- Logistic Regression classifier
- 80/20 train-test split

Outputs accuracy and precision scores after training.
