"""
Trend visualization components for SpectraNLP.
"""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

class TrendPlots:
  """
  Creates visualizations for trend analysis.
  """

  @staticmethod
  def plot_source_comparison(data_sources, title="Sentiment Comparison Across Sources"):
      """
      Create a grouped bar chart comparing sentiment across different data sources.

      Args:
          data_sources (dict): Dictionary mapping source names to DataFrames with sentiment data.
          title (str, optional): Plot title. Defaults to "Sentiment Comparison Across Sources".

      Returns:
          plotly.graph_objects.Figure: Plotly figure object.
      """
      # Prepare data
      combined_data = []

      for source_name, df in data_sources.items():
          if 'sentiment' not in df.columns:
              continue

          sentiment_counts = df['sentiment'].value_counts().reset_index()
          sentiment_counts.columns = ['Sentiment', 'Count']
          sentiment_counts['Source'] = source_name
          sentiment_counts['Percentage'] = sentiment_counts['Count'] / sentiment_counts['Count'].sum() * 100

          combined_data.append(sentiment_counts)

      if not combined_data:
          raise ValueError("No valid sentiment data found in data sources.")

      combined_df = pd.concat(combined_data, ignore_index=True)

      # Create grouped bar chart
      fig = px.bar(
          combined_df,
          x='Source',
          y='Percentage',
          color='Sentiment',
          title=title,
          color_discrete_map={'Positive': '#4CAF50', 'Neutral': '#FFC107', 'Negative': '#F44336'},
          barmode='group',
          text_auto='.1f'
      )

      fig.update_traces(texttemplate='%{text}%', textposition='outside')
      fig.update_layout(
          xaxis_title='Data Source',
          yaxis_title='Percentage',
          legend_title='Sentiment',
          uniformtext_minsize=8,
          uniformtext_mode='hide'
      )

      return fig

  @staticmethod
  def plot_keyword_comparison(df, keywords, text_col='text', title="Keyword Sentiment Comparison"):
      """
      Create a heatmap comparing sentiment across different keywords.

      Args:
          df (pandas.DataFrame): DataFrame with text and sentiment data.
          keywords (list): List of keywords to analyze.
          text_col (str, optional): Name of the text column. Defaults to 'text'.
          title (str, optional): Plot title. Defaults to "Keyword Sentiment Comparison".

      Returns:
          matplotlib.figure.Figure: Matplotlib figure with the heatmap.
      """
      if text_col not in df.columns or 'sentiment' not in df.columns:
          raise ValueError(f"DataFrame must contain '{text_col}' and 'sentiment' columns.")

      # Prepare data
      keyword_sentiment = pd.DataFrame(index=keywords, columns=['Positive', 'Neutral', 'Negative'])

      for keyword in keywords:
          # Filter texts containing the keyword
          keyword_filter = df[text_col].str.contains(keyword, case=False, na=False)
          filtered_df = df[keyword_filter]

          if not filtered_df.empty:
              # Calculate sentiment percentages
              sentiment_counts = filtered_df['sentiment'].value_counts(normalize=True) * 100

              # Fill in the data
              for sentiment in ['Positive', 'Neutral', 'Negative']:
                  keyword_sentiment.at[keyword, sentiment] = sentiment_counts.get(sentiment, 0)
          else:
              # If no matches, set all to 0
              keyword_sentiment.loc[keyword] = [0, 0, 0]

      # Create heatmap
      plt.figure(figsize=(10, 8))
      sns.set(font_scale=1.2)

      # Create heatmap
      ax = sns.heatmap(
          keyword_sentiment,
          annot=True,
          fmt='.1f',
          cmap='RdYlGn',
          linewidths=0.5,
          cbar_kws={'label': 'Percentage (%)'}
      )

      plt.title(title, fontsize=16)
      plt.xlabel('Sentiment', fontsize=14)
      plt.ylabel('Keyword', fontsize=14)

      fig = plt.gcf()
      plt.tight_layout()

      return fig

  @staticmethod
  def plot_sentiment_intensity(df, date_col='date', score_col='sentiment_score', interval='M',
                              title="Sentiment Intensity Over Time"):
      """
      Create a line chart showing sentiment intensity over time.

      Args:
          df (pandas.DataFrame): DataFrame with sentiment score and date data.
          date_col (str, optional): Name of the date column. Defaults to 'date'.
          score_col (str, optional): Name of the sentiment score column. Defaults to 'sentiment_score'.
          interval (str, optional): Time grouping interval ('D' for day, 'W' for week,
              'M' for month, 'Y' for year). Defaults to 'M'.
          title (str, optional): Plot title. Defaults to "Sentiment Intensity Over Time".

      Returns:
          plotly.graph_objects.Figure: Plotly figure object.
      """
      if score_col not in df.columns or date_col not in df.columns:
          raise ValueError(f"DataFrame must contain '{score_col}' and '{date_col}' columns.")

      # Ensure date column is datetime type
      df_copy = df.copy()
      df_copy[date_col] = pd.to_datetime(df_copy[date_col])

      # Group by date
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

      # Calculate average sentiment score per period
      sentiment_intensity = df_copy.groupby('period')[score_col].agg(['mean', 'count']).reset_index()
      sentiment_intensity.columns = ['period', 'avg_score', 'count']

      # Create line chart
      fig = go.Figure()

      # Add main line for average sentiment
      fig.add_trace(go.Scatter(
          x=sentiment_intensity['period'],
          y=sentiment_intensity['avg_score'],
          mode='lines+markers',
          name='Average Sentiment',
          line=dict(color='royalblue', width=3),
          marker=dict(size=8),
      ))

      # Add reference line at 0
      fig.add_shape(
          type="line",
          x0=sentiment_intensity['period'].min(),
          y0=0,
          x1=sentiment_intensity['period'].max(),
          y1=0,
          line=dict(color="gray", width=1, dash="dot"),
      )

      # Add colored regions for positive and negative sentiment
      fig.add_trace(go.Scatter(
          x=sentiment_intensity['period'],
          y=[0] * len(sentiment_intensity),
          fill=None,
          mode='lines',
          line=dict(color='rgba(0,0,0,0)'),
          showlegend=False
      ))

      fig.add_trace(go.Scatter(
          x=sentiment_intensity['period'],
          y=sentiment_intensity['avg_score'],
          fill='tonexty',
          mode='none',
          fillcolor='rgba(76, 175, 80, 0.3)',  # Green for positive
          name='Positive Sentiment Zone'
      ))

      fig.add_trace(go.Scatter(
          x=sentiment_intensity['period'],
          y=[0] * len(sentiment_intensity),
          fill=None,
          mode='lines',
          line=dict(color='rgba(0,0,0,0)'),
          showlegend=False
      ))

      fig.add_trace(go.Scatter(
          x=sentiment_intensity['period'],
          y=[min(0, score) for score in sentiment_intensity['avg_score']],
          fill='tonexty',
          mode='none',
          fillcolor='rgba(244, 67, 54, 0.3)',  # Red for negative
          name='Negative Sentiment Zone'
      ))

      # Update layout
      fig.update_layout(
          title=title,
          xaxis_title='Time Period',
          yaxis_title='Average Sentiment Score',
          yaxis=dict(
              range=[-1, 1],
              tickvals=[-1, -0.5, 0, 0.5, 1],
              ticktext=['-1.0<br>(Very Negative)', '-0.5<br>(Negative)', '0<br>(Neutral)', '0.5<br>(Positive)', '1.0<br>(Very Positive)']
          ),
          legend=dict(
              orientation="h",
              yanchor="bottom",
              y=1.02,
              xanchor="right",
              x=1
          )
      )

      return fig