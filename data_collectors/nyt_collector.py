"""
New York Times API client for fetching articles for sentiment analysis.
"""
import datetime
import time
import pandas as pd
from pynytimes import NYTAPI
import config

class NYTCollector:
    """
    A client for collecting data from the New York Times API.
    """
    def __init__(self, api_key=None):
        """
        Initialize the NYT API client.

        Args:
            api_key (str, optional): NYT API key. Defaults to value from config.
        """
        self.api_key = api_key or config.NYT_API_KEY
        self.nyt = NYTAPI(self.api_key, parse_dates=True)

    def search_articles(self, query, start_date, end_date, max_results=25):
        """
        Search for articles based on query and date range.

        Args:
            query (str): Search term.
            start_date (str or datetime): Start date in 'YYYY-MM-DD' format or as datetime object.
            end_date (str or datetime): End date in 'YYYY-MM-DD' format or as datetime object.
            max_results (int, optional): Maximum number of results to fetch. Defaults to 25.

        Returns:
            list: List of article data.
        """
        # Convert string dates to datetime if needed
        if isinstance(start_date, str):
            start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d')
        if isinstance(end_date, str):
            end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d')

        try:
            articles = self.nyt.article_search(
                query=query,
                results=max_results,
                dates={
                    "begin": start_date,
                    "end": end_date
                },
                options={
                    "sort": "newest"
                }
            )
            return articles
        except Exception as e:
            print(f"Error fetching articles for query '{query}': {e}")
            return []

    def extract_article_data(self, articles):
        """
        Extract relevant data from article objects.

        Args:
            articles (list): List of article objects from NYT API.

        Returns:
            pandas.DataFrame: DataFrame of extracted article data.
        """
        article_data = []

        for article in articles:
            try:
                data = {
                    'headline': article.get('headline', {}).get('main', ''),
                    'lead_paragraph': article.get('lead_paragraph', ''),
                    'abstract': article.get('abstract', ''),
                    'keywords': ', '.join([kw.get('value', '') for kw in article.get('keywords', [])]),
                    'pub_date': article.get('pub_date', ''),
                    'url': article.get('web_url', ''),
                    'source': article.get('source', ''),
                    'document_type': article.get('document_type', ''),
                    'news_desk': article.get('news_desk', ''),
                    'section_name': article.get('section_name', '')
                }
                article_data.append(data)
            except Exception as e:
                print(f"Error processing article: {e}")
                continue

        return pd.DataFrame(article_data)

    def collect_data(self, keywords, start_date, end_date, max_results=25):
        """
        Collect data for a list of keywords within a date range.

        Args:
            keywords (list or str): Search term(s) to look for.
            start_date (str): Start date in 'YYYY-MM-DD' format.
            end_date (str): End date in 'YYYY-MM-DD' format.
            max_results (int, optional): Maximum number of results to fetch per keyword. Defaults to 25.

        Returns:
            pandas.DataFrame: DataFrame of articles.
        """
        if isinstance(keywords, str):
            keywords = [keywords]

        all_articles = []

        for keyword in keywords:
            articles = self.search_articles(keyword, start_date, end_date, max_results)
            all_articles.extend(articles)
            # Respect rate limits
            time.sleep(1)

        # Convert to DataFrame and remove duplicates
        if all_articles:
            df = self.extract_article_data(all_articles)
            # Remove duplicates based on headline
            df = df.drop_duplicates(subset=['headline'])
            return df

        return pd.DataFrame()