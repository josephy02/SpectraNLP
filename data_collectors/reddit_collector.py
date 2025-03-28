"""
Reddit data collector for sentiment analysis.
This module processes pre-downloaded Reddit data as we're not using the live API.
"""
import pandas as pd

class RedditCollector:
    """
    A class for collecting and processing Reddit data.
    Since we're working with pre-downloaded data, this primarily handles file loading
    and data filtering.
    """
    def __init__(self, data_file=None):
        """
        Initialize the Reddit data collector.

        Args:
            data_file (str, optional): Path to the Reddit data file. If None, will look for it
                at runtime.
        """
        self.data_file = data_file
        self.data = None

    def load_data(self, data_file=None):
        """
        Load the Reddit data from a CSV file.

        Args:
            data_file (str, optional): Path to the Reddit data file. Defaults to self.data_file.

        Returns:
            pandas.DataFrame: Loaded data.
        """
        file_path = data_file or self.data_file

        try:
            df = pd.read_csv(file_path)
            self.data = df
            return df
        except Exception as e:
            print(f"Error loading Reddit data: {e}")
            return pd.DataFrame()

    def filter_data(self, start_date=None, end_date=None, keywords=None):
        """
        Filter the Reddit data based on date range and keywords.

        Args:
            start_date (str, optional): Start date in 'YYYY-MM-DD' format.
            end_date (str, optional): End date in 'YYYY-MM-DD' format.
            keywords (list or str, optional): Keyword(s) to filter by.

        Returns:
            pandas.DataFrame: Filtered data.
        """
        if self.data is None:
            print("No data loaded. Call load_data() first.")
            return pd.DataFrame()

        filtered_df = self.data.copy()

        # Convert created_time to datetime if it's not already
        if 'created_time' in filtered_df.columns and not pd.api.types.is_datetime64_any_dtype(filtered_df['created_time']):
            filtered_df['created_time'] = pd.to_datetime(filtered_df['created_time'])

        # Filter by date range
        if start_date and 'created_time' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['created_time'] >= start_date]

        if end_date and 'created_time' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['created_time'] <= end_date]

        # Filter by keywords
        if keywords:
            if isinstance(keywords, str):
                keywords = [keywords]

            # Filter comments containing any of the keywords (case-insensitive)
            if 'self_text' in filtered_df.columns:
                keyword_filter = filtered_df['self_text'].str.contains('|'.join(keywords), case=False, na=False)
                filtered_df = filtered_df[keyword_filter]

        # Extract relevant columns
        relevant_columns = ['self_text', 'created_time']
        available_columns = [col for col in relevant_columns if col in filtered_df.columns]

        return filtered_df[available_columns]

    def collect_data(self, data_file, start_date=None, end_date=None, keywords=None):
        """
        Collect Reddit data based on criteria.

        Args:
            data_file (str): Path to the Reddit data file.
            start_date (str, optional): Start date in 'YYYY-MM-DD' format.
            end_date (str, optional): End date in 'YYYY-MM-DD' format.
            keywords (list or str, optional): Keyword(s) to filter by.

        Returns:
            pandas.DataFrame: Collected and filtered data.
        """
        self.data_file = data_file
        self.load_data()

        if self.data is None or self.data.empty:
            return pd.DataFrame()

        filtered_data = self.filter_data(start_date, end_date, keywords)

        # Standardize column names to match other data sources
        if not filtered_data.empty:
            if 'self_text' in filtered_data.columns:
                filtered_data = filtered_data.rename(columns={'self_text': 'text'})

        return filtered_data