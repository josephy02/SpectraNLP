"""
Flickr API client for fetching photos and comments for sentiment analysis.
"""
import html
import re
import time
from datetime import datetime
import pandas as pd
from flickrapi import FlickrAPI
import config

class FlickrCollector:
    """
    A client for collecting data from Flickr API.
    """
    def __init__(self, api_key=None, api_secret=None):
        """
        Initialize the Flickr API client.

        Args:
            api_key (str, optional): Flickr API key. Defaults to value from config.
            api_secret (str, optional): Flickr API secret. Defaults to value from config.
        """
        self.api_key = api_key or config.FLICKR_API_KEY
        self.api_secret = api_secret or config.FLICKR_API_SECRET
        self.flickr = FlickrAPI(self.api_key, self.api_secret, format="parsed-json")

    def clean_comment_text(self, text):
        """
        Clean and normalize comment text.

        Args:
            text (str): Raw comment text.

        Returns:
            str or None: Cleaned text, or None if text is too short after cleaning.
        """
        text = html.unescape(text)
        text = re.sub(r"https?://\S+", "", text)  # Remove URLs
        text = re.sub(r"<[^>]+>", "", text)  # Remove HTML tags
        text = re.sub(r"[^\x00-\x7F]+", "", text)  # Remove non-ASCII characters
        text = " ".join(text.split()).strip()  # Normalize whitespace
        return text if len(text) > 3 else None

    def search_for_photos(self, keyword, start_date, end_date, num_images=100):
        """
        Search for photos based on keywords and date range.

        Args:
            keyword (str): Search term or tag to look for.
            start_date (str): Start date in 'YYYY-MM-DD' format.
            end_date (str): End date in 'YYYY-MM-DD' format.
            num_images (int, optional): Maximum number of images to fetch. Defaults to 100.

        Returns:
            list: List of photo IDs matching the search criteria.
        """
        start = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp())
        end = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp())

        photos = self.flickr.photos.search(
            tags=keyword,
            tag_mode="all",
            min_upload_date=start,
            max_upload_date=end,
            per_page=num_images,
            sort="date-posted-desc",
            extras="date_upload"
        )

        return [photo['id'] for photo in photos['photos']['photo']]

    def fetch_comments(self, photo_ids):
        """
        Fetch comments for a list of photo IDs.

        Args:
            photo_ids (list): List of Flickr photo IDs.

        Returns:
            tuple: (DataFrame of comments, dict of comment counts per photo)
        """
        comment_data = []
        photo_comment_counts = {}

        for photo_id in photo_ids:
            try:
                response = self.flickr.photos.comments.getList(photo_id=photo_id)
                comments = response.get("comments", {}).get("comment", [])
                photo_comment_counts[photo_id] = len(comments)

                for comment in comments:
                    comment_text = self.clean_comment_text(comment.get("_content", ""))
                    if comment_text:
                        comment_data.append({
                            "photo_id": photo_id,
                            "author": comment.get("authorname", ""),
                            "date": datetime.fromtimestamp(
                                int(comment.get("datecreate", 0))
                            ).strftime("%Y-%m-%d"),
                            "comment_text": comment_text,
                        })

                time.sleep(1)  # Respect API rate limits
            except Exception as e:
                print(f"Error fetching comments for photo ID {photo_id}: {e}")

        return pd.DataFrame(comment_data), photo_comment_counts

    def get_photo_details(self, photo_ids):
        """
        Get photo details for a list of photo IDs.

        Args:
            photo_ids (list): List of Flickr photo IDs.

        Returns:
            list: List of dictionaries with photo details.
        """
        photo_details = []

        for photo_id in photo_ids:
            try:
                info = self.flickr.photos.getInfo(photo_id=photo_id)
                photo = info['photo']

                # Construct photo URL
                farm_id = photo.get('farm')
                server_id = photo.get('server')
                secret = photo.get('secret')
                photo_url = f"https://farm{farm_id}.staticflickr.com/{server_id}/{photo_id}_{secret}.jpg"

                photo_details.append({
                    "id": photo_id,
                    "title": photo.get('title', {}).get('_content', ''),
                    "url": photo_url,
                    "owner": photo.get('owner', {}).get('username', ''),
                    "date_taken": photo.get('dates', {}).get('taken', ''),
                    "tags": [tag.get('_content', '') for tag in photo.get('tags', {}).get('tag', [])]
                })

                time.sleep(1)  # Respect API rate limits
            except Exception as e:
                print(f"Error fetching details for photo ID {photo_id}: {e}")

        return photo_details

    def collect_data(self, keywords, start_date, end_date, num_images=100):
        """
        Collect data for a list of keywords within a date range.

        Args:
            keywords (list or str): Search term(s) or tag(s) to look for.
            start_date (str): Start date in 'YYYY-MM-DD' format.
            end_date (str): End date in 'YYYY-MM-DD' format.
            num_images (int, optional): Maximum number of images to fetch per keyword. Defaults to 100.

        Returns:
            pandas.DataFrame: DataFrame of comments with photo details.
        """
        if isinstance(keywords, str):
            keywords = [keywords]

        all_photo_ids = []
        for keyword in keywords:
            photo_ids = self.search_for_photos(keyword, start_date, end_date, num_images)
            all_photo_ids.extend(photo_ids)

        # Remove duplicates while preserving order
        unique_photo_ids = list(dict.fromkeys(all_photo_ids))

        comments_df, _ = self.fetch_comments(unique_photo_ids)

        if comments_df.empty:
            return pd.DataFrame()

        return comments_df