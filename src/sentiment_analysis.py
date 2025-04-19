from transformers import pipeline

def analyze_sentiment(news_headlines):
    """Analyze sentiment of news headlines."""
    sentiment_pipeline = pipeline("sentiment-analysis")
    sentiments = sentiment_pipeline(news_headlines)
    return np.mean([1 if s['label'] == 'POSITIVE' else -1 for s in sentiments])
