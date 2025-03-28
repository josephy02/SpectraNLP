"""
SpectraNLP - A comprehensive sentiment analysis platform.
This is the main application file for the Streamlit interface.
"""
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from wordcloud import WordCloud
import nltk
import re
import os
import time


from data_collectors.flickr_collector import FlickrCollector
from data_collectors.nyt_collector import NYTCollector
from data_collectors.reddit_collector import RedditCollector
from analysis.sentiment_analyzer import SentimentAnalyzer
from analysis.text_processor import TextProcessor
from visualization.sentiment_plots import SentimentPlots
from visualization.trend_plots import TrendPlots
from utils.helpers import standardize_dataframe, merge_dataframes
import config

# Ensure required NLTK data is downloaded
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('vader_lexicon')
    nltk.download('punkt')
    nltk.download('stopwords')

# Set page configuration
st.set_page_config(
    page_title="SpectraNLP - Sentiment Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cache frequently used data
@st.cache_data(ttl=10)  # Cache for 1 hour
def load_cached_data(source, keywords, start_date, end_date, max_results=100):
    """
    Load data with caching to prevent repeated API calls.
    """
    if source == "Flickr":
        collector = FlickrCollector()
        data = collector.collect_data(keywords, start_date, end_date, max_results)
        return standardize_dataframe(data, source)

    elif source == "NYT":
        collector = NYTCollector()
        data = collector.collect_data(keywords, start_date, end_date, max_results)
        if 'lead_paragraph' in data.columns:
            data = standardize_dataframe(data, source, text_col='lead_paragraph', date_col='pub_date')
        else:
            data = standardize_dataframe(data, source)
        return data

    elif source == "Reddit":
        # For Reddit, we'll use a static file if it exists
        try:
            csv_file = "reddit_comments.csv"  # Default file name
            if os.path.exists(csv_file):
                data = pd.read_csv(csv_file)

                # Filter by date and keywords
                if 'created_time' in data.columns:
                    data['created_time'] = pd.to_datetime(data['created_time'])
                    data = data[(data['created_time'] >= start_date) & (data['created_time'] <= end_date)]

                if keywords and 'text' in data.columns:
                    # Filter for rows containing any of the keywords
                    keyword_pattern = '|'.join(keywords)
                    data = data[data['text'].str.contains(keyword_pattern, case=False, na=False)]

                return standardize_dataframe(data, source, text_col='text', date_col='created_time')
            else:
                st.warning(f"Reddit data file '{csv_file}' not found. Please provide a valid data file.")
                return pd.DataFrame()
        except Exception as e:
            st.error(f"Error loading Reddit data: {e}")
            return pd.DataFrame()

    return pd.DataFrame()

@st.cache_resource
def get_sentiment_analyzer():
    """Get a cached sentiment analyzer instance."""
    return SentimentAnalyzer()

@st.cache_resource
def get_text_processor():
    """Get a cached text processor instance."""
    return TextProcessor()

# App title
st.title("SpectraNLP - Sentiment Analysis Platform")
st.markdown("""
    Analyze sentiment across multiple data sources related to keywords of interest.
    This application processes data from Flickr comments, New York Times articles, and Reddit discussions.
""")

# Sidebar for controls
st.sidebar.header("Settings")

# Data source selection
st.sidebar.subheader("Data Sources")
use_flickr = st.sidebar.checkbox("Flickr", value=True)
use_nyt = st.sidebar.checkbox("New York Times", value=True)
use_reddit = st.sidebar.checkbox("Reddit", value=True)

# Date range selection
st.sidebar.subheader("Date Range")
default_end_date = datetime.now()
default_start_date = default_end_date - timedelta(days=30)

start_date = st.sidebar.date_input(
    "Start Date",
    value=default_start_date,
    max_value=default_end_date
)

end_date = st.sidebar.date_input(
    "End Date",
    value=default_end_date,
    min_value=start_date,
    max_value=default_end_date
)

# Convert to string format for API calls
start_date_str = start_date.strftime('%Y-%m-%d')
end_date_str = end_date.strftime('%Y-%m-%d')

# Keywords selection
st.sidebar.subheader("Keywords")
default_keywords = config.DEFAULT_SEARCH_TERMS
keyword_input = st.sidebar.text_area(
    "Enter keywords (one per line)",
    '\n'.join(default_keywords)
)
keywords = [k.strip() for k in keyword_input.split('\n') if k.strip()]

# Number of results to fetch
max_results = st.sidebar.slider(
    "Maximum results per source",
    min_value=10,
    max_value=500,
    value=100,
    step=10
)

# Analysis button
run_analysis = st.sidebar.button("Run Analysis", type="primary")

# Advanced settings
with st.sidebar.expander("Advanced Settings"):
    preprocess_text = st.checkbox("Preprocess text", value=True)
    show_wordcloud = st.checkbox("Show word clouds", value=True)
    time_interval = st.selectbox(
        "Time grouping",
        options=[("Day", "D"), ("Week", "W"), ("Month", "M"), ("Year", "Y")],
        format_func=lambda x: x[0],
        index=2  # Month as default
    )[1]  # Get the code (D, W, M, Y)

# Display processing status
status_container = st.empty()

if run_analysis:
    # Initialize data containers
    all_data = []
    sources_data = {}
    sentiment_data = pd.DataFrame()

    with status_container.container():
        st.info("Analysis in progress...")
        progress_bar = st.progress(0)

        # Data collection
        step = 0
        num_steps = sum([use_flickr, use_nyt, use_reddit]) * 2  # Collection + Analysis

        # Flickr data
        if use_flickr:
            st.write("Collecting Flickr data...")
            flickr_data = load_cached_data("Flickr", keywords, start_date_str, end_date_str, max_results)
            if not flickr_data.empty:
                all_data.append(flickr_data)
                sources_data["Flickr"] = flickr_data
                st.write(f"✅ Collected {len(flickr_data)} Flickr comments")
            else:
                st.write("⚠️ No Flickr data found")

            step += 1
            progress_bar.progress(step / num_steps)

        # NYT data
        if use_nyt:
            st.write("Collecting New York Times data...")
            nyt_data = load_cached_data("NYT", keywords, start_date_str, end_date_str, max_results)
            if not nyt_data.empty:
                all_data.append(nyt_data)
                sources_data["New York Times"] = nyt_data
                st.write(f"✅ Collected {len(nyt_data)} NYT articles")
            else:
                st.write("⚠️ No NYT data found")

            step += 1
            progress_bar.progress(step / num_steps)

        # Reddit data
        if use_reddit:
            st.write("Collecting Reddit data...")
            reddit_data = load_cached_data("Reddit", keywords, start_date_str, end_date_str, max_results)
            if not reddit_data.empty:
                all_data.append(reddit_data)
                sources_data["Reddit"] = reddit_data
                st.write(f"✅ Collected {len(reddit_data)} Reddit comments")
            else:
                st.write("⚠️ No Reddit data found")

            step += 1
            progress_bar.progress(step / num_steps)

        # Merge and process data
        if all_data:
            combined_data = merge_dataframes(all_data)
            st.write(f"Combined dataset: {len(combined_data)} records")

            # Text preprocessing
            if preprocess_text:
                st.write("Preprocessing text...")
                text_processor = get_text_processor()
                combined_data = text_processor.preprocess_dataframe(combined_data, text_column='text', new_column='processed_text')
                text_col = 'processed_text'
            else:
                text_col = 'text'

            # Sentiment analysis
            sentiment_analyzer = get_sentiment_analyzer()

            for source, data in sources_data.items():
                st.write(f"Analyzing {source} sentiment...")
                if not data.empty:
                    # Use processed text if available
                    if preprocess_text and 'processed_text' in data.columns:
                        analysis_col = 'processed_text'
                    else:
                        analysis_col = 'text'

                    sources_data[source] = sentiment_analyzer.analyze_dataframe(data, text_column=analysis_col)

                step += 1
                progress_bar.progress(step / num_steps)

            # Combine all sentiment-analyzed data
            sentiment_data_list = [df for df in sources_data.values() if not df.empty]
            if sentiment_data_list:
                sentiment_data = merge_dataframes(sentiment_data_list)
                st.write(f"Sentiment analysis complete: {len(sentiment_data)} records analyzed")
            else:
                st.error("No data available for sentiment analysis")

            # Remove progress indicators when done
            progress_bar.empty()
            st.success("Analysis complete!")
            time.sleep(1)  # Brief pause to show completion

    # Clear the status container
    status_container.empty()

    # Display results if data is available
    if not sentiment_data.empty:
        st.header("Sentiment Analysis Results")

        # Overall statistics
        st.subheader("Overall Statistics")
        col1, col2, col3 = st.columns(3)

        with col1:
            total_records = len(sentiment_data)
            st.metric("Total Records", total_records)

        with col2:
            positive_percentage = (sentiment_data['sentiment'] == 'Positive').mean() * 100
            st.metric("Positive Sentiment", f"{positive_percentage:.1f}%")

        with col3:
            negative_percentage = (sentiment_data['sentiment'] == 'Negative').mean() * 100
            st.metric("Negative Sentiment", f"{negative_percentage:.1f}%")

        # Data source breakdown
        st.subheader("Data Source Breakdown")

        source_counts = sentiment_data['source'].value_counts()
        st.bar_chart(source_counts)

        # Sentiment distribution
        st.subheader("Sentiment Distribution")
        try:
            sentiment_dist_fig = SentimentPlots.plot_sentiment_distribution(sentiment_data)
            st.plotly_chart(sentiment_dist_fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error plotting sentiment distribution: {e}")

        # Sentiment over time
        st.subheader("Sentiment Trends Over Time")
        try:
            sentiment_time_fig = SentimentPlots.plot_sentiment_over_time(
                sentiment_data,
                date_col='date',
                interval=time_interval
            )
            st.plotly_chart(sentiment_time_fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error plotting sentiment trends: {e}")

        # Sentiment intensity
        st.subheader("Sentiment Intensity Over Time")
        try:
            sentiment_intensity_fig = TrendPlots.plot_sentiment_intensity(
                sentiment_data,
                date_col='date',
                score_col='sentiment_score',
                interval=time_interval
            )
            st.plotly_chart(sentiment_intensity_fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error plotting sentiment intensity: {e}")

        # Source comparison
        if len(sources_data) > 1:
            st.subheader("Sentiment Comparison Across Sources")
            try:
                source_comparison_fig = TrendPlots.plot_source_comparison(sources_data)
                st.plotly_chart(source_comparison_fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error plotting source comparison: {e}")

        # Keyword comparison
        if len(keywords) > 1:
            st.subheader("Keyword Sentiment Comparison")
            try:
                keyword_comparison_fig = TrendPlots.plot_keyword_comparison(
                    sentiment_data,
                    keywords,
                    text_col=text_col
                )
                st.pyplot(keyword_comparison_fig)
            except Exception as e:
                st.error(f"Error plotting keyword comparison: {e}")

        # Word clouds
        if show_wordcloud:
            st.subheader("Word Clouds by Sentiment")
            col1, col2, col3 = st.columns(3)

            try:
                with col1:
                    st.write("Positive Sentiment")
                    positive_cloud = SentimentPlots.plot_sentiment_wordcloud(
                        sentiment_data,
                        text_col=text_col,
                        sentiment_filter='Positive'
                    )
                    st.pyplot(positive_cloud)

                with col2:
                    st.write("Neutral Sentiment")
                    neutral_cloud = SentimentPlots.plot_sentiment_wordcloud(
                        sentiment_data,
                        text_col=text_col,
                        sentiment_filter='Neutral'
                    )
                    st.pyplot(neutral_cloud)

                with col3:
                    st.write("Negative Sentiment")
                    negative_cloud = SentimentPlots.plot_sentiment_wordcloud(
                        sentiment_data,
                        text_col=text_col,
                        sentiment_filter='Negative'
                    )
                    st.pyplot(negative_cloud)
            except Exception as e:
                st.error(f"Error generating word clouds: {e}")

        # Sample data with highlighted sentiment
        st.subheader("Sample Data with Sentiment Highlights")

        # Sample selection
        num_samples = min(10, len(sentiment_data))
        sample_type = st.radio(
            "Sample type",
            ["Random", "Most Positive", "Most Negative"],
            horizontal=True
        )

        if sample_type == "Random":
            samples = sentiment_data.sample(num_samples)
        elif sample_type == "Most Positive":
            samples = sentiment_data.sort_values('sentiment_score', ascending=False).head(num_samples)
        else:  # Most Negative
            samples = sentiment_data.sort_values('sentiment_score', ascending=True).head(num_samples)

        for i, (_, row) in enumerate(samples.iterrows()):
            sentiment_color = {
                'Positive': '#4CAF50',  # Green
                'Neutral': '#FFC107',   # Amber
                'Negative': '#F44336'   # Red
            }.get(row['sentiment'], '#757575')  # Gray default

            with st.expander(f"Sample {i+1} - {row['source']} ({row['sentiment']})"):
                st.markdown(f"**Date:** {row['date']}")

                # Prepare text with highlighted emotion words
                if 'emotion_words' in row and row['emotion_words']:
                    emotion_words = row['emotion_words'].split(',')
                    text = row['text']

                    # Highlight emotion words
                    for word in sorted(emotion_words, key=len, reverse=True):
                        if word in text:
                            text = text.replace(
                                word,
                                f'<span style="background-color: {sentiment_color}; color: white; padding: 1px 3px; border-radius: 2px;">{word}</span>'
                            )

                    st.markdown(text, unsafe_allow_html=True)
                else:
                    st.write(row['text'])

                # Display sentiment score with gauge
                st.markdown(f"""
                    <div style="text-align: center;">
                        <span style="color: {sentiment_color}; font-weight: bold;">
                            Sentiment Score: {row['sentiment_score']:.2f}
                        </span>
                    </div>
                """, unsafe_allow_html=True)

        # Data explorer
        with st.expander("Data Explorer"):
            st.dataframe(sentiment_data)

            # Allow downloading data
            csv = sentiment_data.to_csv(index=False)
            st.download_button(
                label="Download Data as CSV",
                data=csv,
                file_name="spectranlp_sentiment_data.csv",
                mime="text/csv"
            )
    else:
        if run_analysis:
            st.warning("No data found for the selected sources, keywords, and date range.")

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center;">
        <p>SpectraNLP - Sentiment Analysis Platform | Source code available on <a href="https://github.com/josephy02/SpectraNLP">GitHub</a></p>
    </div>
""", unsafe_allow_html=True)

if __name__ == "__main__":
    # This block will be executed when the script is run directly
    pass