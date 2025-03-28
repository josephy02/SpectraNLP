"""
Configuration module for SpectraNLP.
Handles loading environment variables and configuration settings.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

# Flickr API settings
FLICKR_API_KEY = os.getenv('FLICKR_API_KEY', '')
FLICKR_API_SECRET = os.getenv('FLICKR_API_SECRET', '')

# NYT API settings
NYT_API_KEY = os.getenv('NYT_API_KEY', '')

# Default search settings
DEFAULT_SEARCH_TERMS = [
    'gaza',
    'palestine',
    'israel',
    'hamas',
    'palestinian refugees',
    'israel palestine conflict'
]

# Date range settings
DEFAULT_START_DATE = "2023-01-01"
DEFAULT_END_DATE = "2024-11-01"

# Sentiment analysis settings
VADER_CUSTOM_LEXICON = {
    # Negative sentiment terms
    'casualty': -0.6,
    'casualties': -0.6,
    'death': -0.8,
    'deaths': -0.8,
    'killed': -0.8,
    'killing': -0.8,
    'injured': -0.6,
    'wound': -0.6,
    'wounded': -0.6,
    'displaced': -0.5,
    'displacement': -0.5,
    'refugee': -0.4,
    'refugees': -0.4,
    'destruction': -0.7,
    'destroyed': -0.7,
    'damage': -0.5,
    'damaged': -0.5,
    'crisis': -0.6,
    'conflict': -0.4,
    'violence': -0.7,
    'violent': -0.7,
    'attack': -0.6,
    'attacks': -0.6,
    'siege': -0.6,
    'blockade': -0.5,
    'suffering': -0.7,
    'hostage': -0.8,
    'hostages': -0.8,

    # Positive sentiment terms
    'peace': 0.8,
    'peaceful': 0.7,
    'ceasefire': 0.6,
    'truce': 0.6,
    'negotiation': 0.5,
    'negotiations': 0.5,
    'diplomatic': 0.5,
    'diplomacy': 0.5,
    'agreement': 0.6,
    'resolution': 0.6,
    'dialogue': 0.6,
    'humanitarian': 0.5,
    'aid': 0.6,
    'assistance': 0.5,
    'support': 0.4,
    'relief': 0.5,
    'reconciliation': 0.7,
    'stability': 0.6,
    'stable': 0.5,
    'protect': 0.5,
    'protection': 0.5,
    'safety': 0.6,
    'safe': 0.6,
    'rebuild': 0.5,
    'rebuilding': 0.5,
}