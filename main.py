# %%
from googleapiclient.discovery import build
import pandas as pd
import nltk
from textblob import TextBlob
import numpy as np
from wordcloud import WordCloud
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import re
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score






# %%
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


# %%
stop_words = set(stopwords.words("english"))
lemmatizer = nltk.WordNetLemmatizer()




# %%
API_KEY = 'API_KEY'






# %%
def vid_Comment(video_id, max_results=10000):
    
    Comment = []
    
    n = len(Comment)
    
    youtube = build('youtube', 'v3', developerKey=API_KEY)
    
    next_page = None
    
    while len(Comment) < max_results:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=min(100, max_results - len(Comment)),
            pageToken = next_page
        )
        
        response = request.execute()
        
        for item in response["items"]:
            comment = item["snippet"]['topLevelComment']['snippet']['textDisplay']
            Comment.append(comment)
            
        next_page = response.get('nextPageToken')
        
        if not next_page:
            break
        
    return pd.DataFrame(Comment, columns = ["Comment"])






# %%
video_id = "DOgnE6JQqso"
df = vid_Comment(video_id)





# %%
def preprocess(text):
    if not isinstance(text, str):
        return ""
    
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, )
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    text = " ".join([lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words])
    
    return text






# %%
video_id = "DOgnE6JQqso"
df = vid_Comment(video_id)






# %%
df["Cleaned_Comments"] = df["Comment"].apply(preprocess)  
print(df["Cleaned_Comments"])




    
# %%
def get_sentiment(text):
    analysis = TextBlob(text).sentiment.polarity
    return 1 if analysis > 0 else 0 
df["Sentiment"] = df["Cleaned_Comments"].apply(get_sentiment)

x_train, x_test, y_train, y_test = train_test_split(df["Cleaned_Comments"], df["Sentiment"], test_size = 0.2, random_state = 42)




# %%
vectorizer = TfidfVectorizer(max_features=333)
x_train_tif = vectorizer.fit_transform(x_train)
x_test_tif = vectorizer.transform(x_test)



# %%
model = LogisticRegression()
model.fit(x_train_tif, y_train)


# %%
y_prediction = model.predict(x_test_tif)



# %%
accuracy = accuracy_score(y_test, y_prediction)
precision = precision_score(y_test, y_prediction)
print(f"Model Accuracy: {accuracy:.2f}")
print(f"Precision : {precision:.2f}")




# %%
def predict_sentiment(new_comment):
    cleaned_comment = preprocess(new_comment)
    vectorized_comment = vectorizer.transform([cleaned_comment])
    prediction = model.predict(vectorized_comment)
    return "positive" if prediction == 1 else "negative"



# %%
print(predict_sentiment(" hard see car getting damaged"))
