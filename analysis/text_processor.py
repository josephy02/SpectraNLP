"""
Text preprocessing utilities for SpectraNLP.
"""
import re
import unicodedata
import nltk
import contractions
import inflect
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

class TextProcessor:
  """
  Process and clean text data for analysis.
  """
  def __init__(self):
      """
      Initialize the text processor.
      """
      # Download required NLTK data
      try:
          nltk.data.find('tokenizers/punkt')
          nltk.data.find('corpora/stopwords')
          nltk.data.find('corpora/wordnet')
      except LookupError:
          nltk.download('punkt')
          nltk.download('stopwords')
          nltk.download('wordnet')

      self.lemmatizer = WordNetLemmatizer()
      self.p_engine = inflect.engine()

  def preprocess_text(self, text, lemmatize=True, remove_stopwords=True):
      """
      Preprocess text by applying multiple cleaning steps.

      Args:
          text (str): Text to preprocess.
          lemmatize (bool, optional): Whether to apply lemmatization. Defaults to True.
          remove_stopwords (bool, optional): Whether to remove stopwords. Defaults to True.

      Returns:
          str: Preprocessed text.
      """
      if not text:
          return ""

      # Convert to string if not already
      text = str(text)

      # Remove URLs
      text = self.remove_urls(text)

      # Replace contractions (e.g., "don't" -> "do not")
      text = self.replace_contractions(text)

      # Tokenize
      tokens = nltk.word_tokenize(text)

      # Remove non-ASCII characters
      tokens = self.remove_non_ascii(tokens)

      # Convert to lowercase
      tokens = self.to_lowercase(tokens)

      # Remove punctuation
      tokens = self.remove_punctuation(tokens)

      # Replace numbers with text representation
      tokens = self.replace_numbers(tokens)

      # Remove stopwords if requested
      if remove_stopwords:
          tokens = self.remove_stopwords(tokens)

      # Apply lemmatization if requested
      if lemmatize:
          tokens = self.lemmatize_words(tokens)

      # Join tokens back into text
      return " ".join(tokens)

  def remove_urls(self, text):
      """Remove URLs from text."""
      return re.sub(r"http\S+|www\S+|https\S+", "", text)

  def replace_contractions(self, text):
      """Replace contractions in text."""
      return contractions.fix(text)

  def remove_non_ascii(self, tokens):
      """Remove non-ASCII characters from tokens."""
      return [unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore') for word in tokens]

  def to_lowercase(self, tokens):
      """Convert tokens to lowercase."""
      return [word.lower() for word in tokens]

  def remove_punctuation(self, tokens):
      """Remove punctuation from tokens."""
      return [re.sub(r'[^\w\s]', '', word) for word in tokens if re.sub(r'[^\w\s]', '', word)]

  def replace_numbers(self, tokens):
      """Replace numbers with text representation."""
      result = []
      for word in tokens:
          if word.isdigit():
              try:
                  result.append(self.p_engine.number_to_words(word))
              except:
                  result.append(word)
          else:
              result.append(word)
      return result

  def remove_stopwords(self, tokens):
      """Remove stopwords from tokens."""
      return [word for word in tokens if word not in stopwords.words('english')]

  def lemmatize_words(self, tokens):
      """Lemmatize tokens."""
      return [self.lemmatizer.lemmatize(word, pos='v') for word in tokens]

  def preprocess_dataframe(self, df, text_column='text', new_column=None):
      """
      Preprocess text in a DataFrame.

      Args:
          df (pandas.DataFrame): DataFrame containing text data.
          text_column (str, optional): Name of the column containing text.
              Defaults to 'text'.
          new_column (str, optional): Name of the new column to store preprocessed text.
              If None, overwrites the original column. Defaults to None.

      Returns:
          pandas.DataFrame: DataFrame with preprocessed text.
      """
      if df.empty or text_column not in df.columns:
          print(f"Column '{text_column}' not found in DataFrame or DataFrame is empty.")
          return df

      # Create a copy to avoid modifying the original
      result_df = df.copy()

      # Determine output column
      output_column = new_column or text_column

      # Apply preprocessing to each row
      result_df[output_column] = result_df[text_column].apply(self.preprocess_text)

      return result_df