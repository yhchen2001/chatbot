from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
import pandas as pd
from dotenv import load_dotenv
import requests
from typing import List
import time
import sys
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=openai_api_key)

def get_text_chunks(text, chunk_size = 500):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    chunks = [chunk.replace('\n', "") for chunk in chunks]
    return chunks


def get_embeddings(texts: List[str]):
    response = client.embeddings.create(
        input=texts,
        model="text-embedding-ada-002"
    )
    embeddings = [item.embedding for item in response.data]
    return embeddings