"""
Helper utilities for SpectraNLP.
"""
import pandas as pd
import re
from datetime import datetime
import html

def standardize_dataframe(df, source, text_col=None, date_col=None):
    """
    Standardize DataFrame columns for consistent processing.

    Args:
        df (pandas.DataFrame): Source DataFrame.
        source (str): Name of the data source.
        text_col (str, optional): Name of the text column. If None, tries to determine automatically.
        date_col (str, optional): Name of the date column. If None, tries to determine automatically.

    Returns:
        pandas.DataFrame: Standardized DataFrame.
    """
    if df.empty:
        return pd.DataFrame(columns=['text', 'date', 'source'])

    result = df.copy()

    # Add source column
    result['source'] = source

    # Determine text column
    if text_col:
        text_column = text_col
    elif 'comment_text' in df.columns:
        text_column = 'comment_text'
    elif 'self_text' in df.columns:
        text_column = 'self_text'
    elif 'lead_paragraph' in df.columns:
        text_column = 'lead_paragraph'
    elif 'text' in df.columns:
        text_column = 'text'
    else:
        # Look for column that likely contains text
        text_columns = [col for col in df.columns if 'text' in col.lower() or 'comment' in col.lower()]
        text_column = text_columns[0] if text_columns else None

    # Determine date column
    if date_col:
        date_column = date_col
    elif 'date' in df.columns:
        date_column = 'date'
    elif 'created_time' in df.columns:
        date_column = 'created_time'
    elif 'pub_date' in df.columns:
        date_column = 'pub_date'
    else:
        # Look for column that likely contains dates
        date_columns = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
        date_column = date_columns[0] if date_columns else None

    # Standardize text column
    if text_column and text_column in df.columns:
        result.rename(columns={text_column: 'text'}, inplace=True)
    elif 'text' not in result.columns:
        # If we couldn't find a text column, create an empty one
        result['text'] = ''

    # Standardize date column
    if date_column and date_column in df.columns:
        result.rename(columns={date_column: 'date'}, inplace=True)

        # Convert to datetime
        if not pd.api.types.is_datetime64_any_dtype(result['date']):
            try:
                result['date'] = pd.to_datetime(result['date'])
            except:
                pass
    elif 'date' not in result.columns:
        # If we couldn't find a date column, use current date
        result['date'] = datetime.now()

    # Ensure date is in consistent format
    if 'date' in result.columns:
        try:
            result['date'] = pd.to_datetime(result['date'])
        except:
            result['date'] = datetime.now()

    # Keep only necessary columns
    required_columns = ['text', 'date', 'source']
    optional_columns = ['author', 'sentiment', 'sentiment_score', 'emotion_words']
    available_columns = required_columns + [col for col in optional_columns if col in result.columns]

    return result[available_columns]

def clean_html(html_content):
    """
    Clean HTML content by removing tags but preserving structure.

    Args:
        html_content (str): HTML content to clean.

    Returns:
        str: Cleaned text.
    """
    if not html_content:
        return ""

    # Unescape HTML entities
    text = html.unescape(html_content)

    # Replace common tags with newlines or spaces to preserve structure
    text = re.sub(r'<br\s*/?>', '\n', text)
    text = re.sub(r'<p\s*/?>', '\n\n', text)
    text = re.sub(r'</p>', '', text)
    text = re.sub(r'<div\s*/?>', '\n', text)
    text = re.sub(r'</div>', '', text)
    text = re.sub(r'<li\s*/?>', '\n• ', text)
    text = re.sub(r'</li>', '', text)
    text = re.sub(r'<ul\s*/?>', '\n', text)
    text = re.sub(r'</ul>', '\n', text)

    # Remove all remaining HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)

    # Fix whitespace
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n\s*\n', '\n\n', text)

    return text.strip()

def extract_keywords(text, num_keywords=5, min_length=3):
    """
    Extract potential keywords from text based on frequency.

    Args:
        text (str): Text to extract keywords from.
        num_keywords (int, optional): Number of keywords to extract. Defaults to 5.
        min_length (int, optional): Minimum keyword length. Defaults to 3.

    Returns:
        list: List of extracted keywords.
    """
    if not text:
        return []

    # Tokenize and clean
    words = re.findall(r'\b[a-zA-Z]{' + str(min_length) + r',}\b', text.lower())

    # Count word frequency
    word_counts = {}
    for word in words:
        if word not in word_counts:
            word_counts[word] = 0
        word_counts[word] += 1

    # Sort by frequency
    keywords = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)

    # Return top keywords
    return [word for word, count in keywords[:num_keywords]]

def merge_dataframes(df_list, preserve_columns=True):
    """
    Merge multiple DataFrames into one, preserving common columns.

    Args:
        df_list (list): List of DataFrames to merge.
        preserve_columns (bool, optional): Whether to preserve all columns.
            If False, keeps only common columns. Defaults to True.

    Returns:
        pandas.DataFrame: Merged DataFrame.
    """
    if not df_list:
        return pd.DataFrame()

    if len(df_list) == 1:
        return df_list[0]

    if preserve_columns:
        return pd.concat(df_list, ignore_index=True)
    else:
        # Find common columns
        common_columns = set(df_list[0].columns)
        for df in df_list[1:]:
            common_columns = common_columns.intersection(set(df.columns))

        # Ensure at least text, date, and source columns
        required_columns = ['text', 'date', 'source']
        for col in required_columns:
            if col not in common_columns:
                common_columns.add(col)

        # Merge with common columns
        result_dfs = []
        for df in df_list:
            available_columns = [col for col in common_columns if col in df.columns]
            result_dfs.append(df[available_columns])

        return pd.concat(result_dfs, ignore_index=True)