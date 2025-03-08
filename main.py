# %%
from googleapiclient.discovery import build
import pandas as pd
import nltk
from textblob import TextBlob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# %%
nltk.download('punkt')

# %%
API_KEY = 'AIzaSyBDTtiskIRokwlJaJTFaQeI7ymJPGtH_SI'

# %%
def vid_comments(video_id, max_results=100):
    comments = []
    n = len(comments)
    youtube = build('youtube', 'v3', developerKey=API_KEY)
    next_page = None
    while len(comments) < max_results:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=min(100, max_results - len(comments)),
            pageToken = next_page
        )
        response = request.execute()
        for item in response["items"]:
            comment = item["snippet"]['topLevelComment']['snippet']['textDisplay']
            comments.append(comment)
        next_page = response.get('nextPageToken')
        if not next_page:
            break
    return pd.DataFrame(comments, columns = ["comments"])

# %%
video_id = "DOgnE6JQqso"
df = vid_comments(video_id)

# %%
print(df.head)

# %%
def analyze_sentiment(comment):
    """Classify sentiment as Positive, Negative, or Neutral using TextBlob."""
    analysis = TextBlob(comment)
    polarity = analysis.sentiment.polarity
    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    else:
        return "Neutral"
    
# %%
def analyze_comments(df):
    """Analyze sentiment distribution of YouTube comments."""
    df["Sentiment"] = df["Comment"].apply(analyze_sentiment)
    sentiment_counts = df["Sentiment"].value_counts()
    plt.figure(figsize=(8, 5))
    sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette=["green", "red", "gray"])
    plt.title("Sentiment Distribution")
    plt.xlabel("Sentiment")
    plt.ylabel("Number of Comments")
    plt.show()

    return df

# %%
video_id = "DOgnE6JQqso"
df = vid_comments(video_id)
df = analyze_comments(df)
print(df.head())

# %%
def generate_wordcloud(df):
    """Generate a word cloud from the comments."""
    text = " ".join(comment for comment in df["Comment"])
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()


generate_wordcloud(df)
