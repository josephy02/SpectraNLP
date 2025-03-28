# SpectraNLP

SpectraNLP is a comprehensive sentiment analysis platform that analyzes textual data across multiple social media platforms and news sources. It provides insights into public sentiment trends over time on topics of interest, with a particular focus on the Israel-Palestine conflict.

## Features

- **Multi-source Data Collection**: Gather data from Flickr comments, New York Times articles, and Reddit discussions
- **Sentiment Analysis**: Analyze the emotional tone of content using VADER sentiment analysis
- **Temporal Tracking**: Monitor sentiment changes over time through interactive visualizations
- **Topic Filtering**: Filter data by specific keywords and date ranges
- **Interactive Dashboard**: Explore data through a user-friendly Streamlit web interface

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/josephy02/SpectraNLP.git
   cd SpectraNLP
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the project root and add your API keys:
   ```
   FLICKR_API_KEY=your_flickr_api_key
   FLICKR_API_SECRET=your_flickr_api_secret
   NYT_API_KEY=your_nyt_api_key
   ```

## Usage

### Running the Streamlit App

```bash
streamlit run app.py
```

This will start the Streamlit server and open the dashboard in your default web browser.

### Dashboard Features

- **Source Selection**: Choose data sources to analyze (Flickr, NYT, Reddit)
- **Keyword Search**: Enter terms related to your topic of interest
- **Date Range**: Select the time period for analysis
- **Visualization Types**:
  - Sentiment distribution
  - Sentiment trends over time
  - Word clouds for positive/negative sentiment
  - Comment highlights with sentiment indicators

## Project Structure

```
SpectraNLP/
├── app.py                     # Main Streamlit application
├── data_collectors/           # API clients and data collection modules
├── analysis/                  # Data analysis modules
├── visualization/             # Visualization components
└── utils/                     # Utility functions
```

## Data Sources

- **Flickr**: Comments on photos related to specified keywords
- **New York Times**: Article headlines and content through the NYT API
- **Reddit**: Comments from relevant subreddits

## Methodology

SpectraNLP uses the VADER (Valence Aware Dictionary and sEntiment Reasoner) lexicon for sentiment analysis, which is specifically attuned to sentiments expressed in social media. The sentiment scores range from -1 (extremely negative) to +1 (extremely positive).

For text preprocessing, we:
1. Remove URLs and non-ASCII characters
2. Expand contractions
3. Convert text to lowercase
4. Remove punctuation and stopwords
5. Apply lemmatization for word normalization

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [NLTK](https://www.nltk.org/) for natural language processing tools
- [VADER Sentiment Analysis](https://github.com/cjhutto/vaderSentiment) for sentiment analysis
- [Streamlit](https://streamlit.io/) for the interactive dashboard framework
- [Flickr API](https://www.flickr.com/services/api/), [NYT API](https://developer.nytimes.com/), and Reddit for data sources
