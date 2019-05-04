import pandas as pd
import re
from textblob import TextBlob
df = pd.read_csv("../concordData/tweetTrain.csv", encoding = "ISO-8859-1")
df = df.sample(frac=1)
def polarity(content):
    """Does textblod analysis
    """
    blob = TextBlob(content)
    return blob.sentiment.polarity

clean_tweets = [re.sub(r'@*|https','', tweet) for tweet in df.iloc[:100,-1]]

scores = [polarity(tweet) for tweet in clean_tweets]
result = 0
for score, label in zip(scores, df.iloc[:100, 0]):
    if((score<0 and label==0) or (score>0 and label==4)):
        result += 1

print(result)
