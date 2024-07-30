import pandas as pd
import numpy as np
import re

def preprocess_text(text):
  """
  This function preprocesses the text by removing unwanted characters and symbols.
  """
  if not isinstance(text, str):
    text = str(text)  # Convert non-string values to string

  # Remove URLs
  text = re.sub(r'http\S+', ' ', text)
  # Remove usernames
  text = re.sub(r'@\S+', ' ', text)
  # Remove hashtags
  text = re.sub(r'#\S+', '', text)

  # Remove multiple spaces
  text = re.sub(r'\s+', ' ', text)
  # Convert text to lowercase
  text = text.lower()
  return text




