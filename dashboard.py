import pandas as pd
import numpy as np

from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma

import streamlit

load_dotenv()

df_books = pd.read_csv("books_with_emotions.csv")
