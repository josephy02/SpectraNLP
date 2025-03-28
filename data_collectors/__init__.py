"""
Data collection package for SpectraNLP.
Provides interfaces for various data sources.
"""
from .flickr_collector import FlickrCollector
from .nyt_collector import NYTCollector
from .reddit_collector import RedditCollector

__all__ = ['FlickrCollector', 'NYTCollector', 'RedditCollector']