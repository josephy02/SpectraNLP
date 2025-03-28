"""
Sentiment visualization components for SpectraNLP.
"""
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import numpy as np

class SentimentPlots:
  """
  Creates visualizations for sentiment analysis results.
  """

  @staticmethod
  def plot_sentiment_distribution(df, title="Sentiment Distribution"):
      """
      Create a bar chart of sentiment distribution.

      Args:
          df (pandas.DataFrame): DataFrame with a 'sentiment' column.
          title (str, optional): Plot title. Defaults to "Sentiment Distribution".

      Returns:
          plotly.graph_objects.Figure: Plotly figure object.
      """
      if 'sentiment' not in df.columns:
          raise ValueError("DataFrame must contain a 'sentiment' column.")

      # Count sentiments
      sentiment_counts = df['sentiment'].value_counts().reset_index()
      sentiment_counts.columns = ['Sentiment', 'Count']

      # Set color map
      color_map = {'Positive': '#4CAF50', 'Neutral': '#FFC107', 'Negative': '#F44336'}

      # Create bar chart
      fig = px.bar(
          sentiment_counts,
          x='Sentiment',
          y='Count',
          title=title,
          color='Sentiment',
          color_discrete_map=color_map,
          text='Count'
      )

      fig.update_traces(texttemplate='%{text}', textposition='outside')
      fig.update_layout(
          xaxis_title='Sentiment',
          yaxis_title='Count',
          uniformtext_minsize=8,
          uniformtext_mode='hide'
      )

      return fig

  @staticmethod
  def plot_sentiment_over_time(df, date_col='date', interval='M', title="Sentiment Trends Over Time"):
      """
      Create a line chart of sentiment trends over time.

      Args:
          df (pandas.DataFrame): DataFrame with 'sentiment' and date columns.
          date_col (str, optional): Name of the date column. Defaults to 'date'.
          interval (str, optional): Time grouping interval ('D' for day, 'W' for week,
              'M' for month, 'Y' for year). Defaults to 'M'.
          title (str, optional): Plot title. Defaults to "Sentiment Trends Over Time".

      Returns:
          plotly.graph_objects.Figure: Plotly figure object.
      """
      if 'sentiment' not in df.columns or date_col not in df.columns:
          raise ValueError(f"DataFrame must contain 'sentiment' and '{date_col}' columns.")

      # Ensure date column is datetime type
      df_copy = df.copy()
      df_copy[date_col] = pd.to_datetime(df_copy[date_col])

      # Group by date and sentiment
      if interval == 'D':
          df_copy['period'] = df_copy[date_col].dt.date
      elif interval == 'W':
          df_copy['period'] = df_copy[date_col].dt.to_period('W').dt.start_time.dt.date
      elif interval == 'M':
          df_copy['period'] = df_copy[date_col].dt.to_period('M').dt.start_time.dt.date
      elif interval == 'Y':
          df_copy['period'] = df_copy[date_col].dt.year
      else:
          raise ValueError("Interval must be one of: 'D', 'W', 'M', 'Y'")

      # Count sentiments by period
      sentiment_over_time = df_copy.groupby(['period', 'sentiment']).size().reset_index(name='count')

      # Create line chart
      fig = px.line(
          sentiment_over_time,
          x='period',
          y='count',
          color='sentiment',
          title=title,
          color_discrete_map={'Positive': '#4CAF50', 'Neutral': '#FFC107', 'Negative': '#F44336'}
      )

      # Add markers
      fig.update_traces(mode='lines+markers')

      fig.update_layout(
          xaxis_title='Time Period',
          yaxis_title='Count',
          legend_title='Sentiment'
      )

      return fig

  @staticmethod
  def plot_sentiment_wordcloud(df, text_col='text', sentiment_filter=None, title=None, max_words=100):
      """
      Create a word cloud from text data, optionally filtered by sentiment.

      Args:
          df (pandas.DataFrame): DataFrame with text data.
          text_col (str, optional): Name of the text column. Defaults to 'text'.
          sentiment_filter (str, optional): Filter for specific sentiment ('Positive',
              'Neutral', or 'Negative'). If None, uses all text. Defaults to None.
          title (str, optional): Plot title. If None, generates based on sentiment_filter.
              Defaults to None.
          max_words (int, optional): Maximum number of words in the cloud. Defaults to 100.

      Returns:
          matplotlib.figure.Figure: Matplotlib figure with the word cloud.
      """
      if text_col not in df.columns:
          raise ValueError(f"DataFrame must contain a '{text_col}' column.")

      # Filter by sentiment if specified
      if sentiment_filter and 'sentiment' in df.columns:
          filtered_df = df[df['sentiment'] == sentiment_filter]
          if filtered_df.empty:
              raise ValueError(f"No text found with sentiment '{sentiment_filter}'.")
      else:
          filtered_df = df

      # Combine all text
      all_text = ' '.join(filtered_df[text_col].dropna().astype(str))

      if not all_text:
          raise ValueError("No valid text found for word cloud generation.")

      # Generate word cloud
      wordcloud = WordCloud(
          width=800,
          height=400,
          max_words=max_words,
          background_color='white',
          colormap='viridis',
          contour_width=1,
          contour_color='steelblue'
      ).generate(all_text)

      # Create figure
      fig, ax = plt.subplots(figsize=(10, 6))
      ax.imshow(wordcloud, interpolation='bilinear')
      ax.axis('off')

      # Set title
      if title:
          ax.set_title(title, fontsize=16)
      elif sentiment_filter:
          ax.set_title(f"{sentiment_filter} Sentiment Word Cloud", fontsize=16)
      else:
          ax.set_title("Word Cloud", fontsize=16)

      fig.tight_layout()
      return fig

  @staticmethod
  def highlight_sentiment_text(text, emotion_words, sentiment):
      """
      Highlight emotion words in text with corresponding sentiment color.

      Args:
          text (str): Original text.
          emotion_words (list): List of emotion words to highlight.
          sentiment (str): Sentiment classification ('Positive', 'Neutral', or 'Negative').

      Returns:
          str: HTML-formatted text with highlighted emotion words.
      """
      if not text or not emotion_words:
          return text

      # Set color based on sentiment
      if sentiment == 'Positive':
          color = '#4CAF50'  # Green
      elif sentiment == 'Negative':
          color = '#F44336'  # Red
      else:
          color = '#FFC107'  # Yellow/Amber

      highlighted_text = text

      # Highlight each emotion word
      for word in sorted(emotion_words, key=len, reverse=True):
          if word in highlighted_text:
              highlighted_text = highlighted_text.replace(
                  word,
                  f'<span style="background-color: {color}; padding: 1px 3px; border-radius: 2px;">{word}</span>'
              )

      return highlighted_text