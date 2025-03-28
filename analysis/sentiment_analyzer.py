"""
Sentiment analysis functionality for SpectraNLP.
Uses VADER sentiment analysis with customized lexicon for topic relevance.
"""
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import config

class SentimentAnalyzer:
    """
    Analyzes sentiment of text data using VADER sentiment analysis.
    """
    def __init__(self, custom_lexicon=None):
        """
        Initialize the sentiment analyzer.

        Args:
            custom_lexicon (dict, optional): Custom lexicon to augment VADER.
                Defaults to the lexicon in config.
        """
        # Download VADER lexicon if not already present
        try:
            nltk.data.find('sentiment/vader_lexicon.zip')
        except LookupError:
            nltk.download('vader_lexicon')

        self.sid = SentimentIntensityAnalyzer()

        # Update with custom lexicon
        custom_lex = custom_lexicon or config.VADER_CUSTOM_LEXICON
        if custom_lex:
            self.sid.lexicon.update(custom_lex)

    def analyze_text(self, text):
        """
        Analyze the sentiment of a single text string.

        Args:
            text (str): Text to analyze.

        Returns:
            tuple: (sentiment_label, compound_score, scores_dict)
        """
        if not text or pd.isna(text):
            return 'Neutral', 0.0, {'compound': 0.0, 'neg': 0.0, 'neu': 1.0, 'pos': 0.0}

        scores = self.sid.polarity_scores(str(text))
        compound = scores['compound']

        # Determine sentiment label based on compound score
        if compound >= 0.05:
            sentiment = 'Positive'
        elif compound <= -0.05:
            sentiment = 'Negative'
        else:
            sentiment = 'Neutral'

        return sentiment, compound, scores

    def find_emotion_words(self, text):
        """
        Find words in the text that contribute to sentiment.

        Args:
            text (str): Text to analyze.

        Returns:
            list: List of words that appear in the VADER lexicon.
        """
        if not text or pd.isna(text):
            return []

        words = str(text).lower().split()
        emotion_words = []

        for word in words:
            if word.lower() in self.sid.lexicon:
                emotion_words.append(word)

        return emotion_words

    def analyze_dataframe(self, df, text_column='text'):
        """
        Analyze sentiment for all texts in a DataFrame.

        Args:
            df (pandas.DataFrame): DataFrame containing text data.
            text_column (str, optional): Name of the column containing text.
                Defaults to 'text'.

        Returns:
            pandas.DataFrame: Original DataFrame with added sentiment columns.
        """
        if df.empty or text_column not in df.columns:
            print(f"Column '{text_column}' not found in DataFrame or DataFrame is empty.")
            return df

        # Create a copy to avoid modifying the original
        result_df = df.copy()

        # Apply sentiment analysis to each row
        result_df['sentiment'] = ''
        result_df['sentiment_score'] = 0.0
        result_df['pos_score'] = 0.0
        result_df['neu_score'] = 0.0
        result_df['neg_score'] = 0.0
        result_df['emotion_words'] = None

        for idx, row in result_df.iterrows():
            text = row[text_column]
            sentiment, compound, scores = self.analyze_text(text)
            emotion_words = self.find_emotion_words(text)

            result_df.at[idx, 'sentiment'] = sentiment
            result_df.at[idx, 'sentiment_score'] = compound
            result_df.at[idx, 'pos_score'] = scores['pos']
            result_df.at[idx, 'neu_score'] = scores['neu']
            result_df.at[idx, 'neg_score'] = scores['neg']
            result_df.at[idx, 'emotion_words'] = ','.join(emotion_words) if emotion_words else ''

        return result_df