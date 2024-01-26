
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
from utils import get_embeddings, get_text_chunks

def pdf_to_df(file_path):
    load_dotenv()
    openai_api_key = os.getenv('OPENAI_API_KEY')
    client = OpenAI(api_key=openai_api_key)


    def get_pdf_text(file_name):
        print(file_name)
        reader = PdfReader(file_name)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text
    
    def prompt_seperately(chunks, batch_size = 2000):
        crr_head = 0
        all_embeddings = []
        print(len(chunks))
        while(crr_head < len(chunks)):
            print("crr head = ", crr_head)
            if crr_head + batch_size < len(chunks):
                embeddings = get_embeddings(chunks[crr_head: crr_head + batch_size])
            else:
                embeddings = get_embeddings(chunks[crr_head:])
            all_embeddings += embeddings
            crr_head += batch_size
            if crr_head >= len(chunks):
                break
            time.sleep(61)
            sys.stdout.flush()
        return all_embeddings

    def get_embedding_df(text):
        print(len(text))
        chunks = get_text_chunks(text, chunk_size=500)
        df = pd.DataFrame({'text': chunks})
        embeddings = prompt_seperately(chunks)
        df['embeddings'] = embeddings
        return df

    text = get_pdf_text(file_path)
    df = get_embedding_df(text)

    return df


if __name__ == '__main__' :
    pdf_file_dir = 'pdfs' #
    dfs = pdf_to_df(pdf_file_dir)
    for df in dfs:
        print(df)
