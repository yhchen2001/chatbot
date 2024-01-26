
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

#global settings
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=openai_api_key)

pdf_file_dir = 'pdfs' #

def pdfs_to_dfs(directory):

    def get_pdf_text(file_name):
        print(file_name)
        reader = PdfReader(file_name)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text

    def get_text_chunks(text, chunk_size = 2000):
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=chunk_size,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        return chunks
    

    def get_embeddings(texts: List[str]):
        response = client.embeddings.create(
            input=texts,
            model="text-embedding-ada-002"
        )
        embeddings = [item.embedding for item in response.data]
        return embeddings
    
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
    
    dfs = []
    for file_name in os.listdir(directory):
        file_path = os.path.join(directory, file_name)
    
        # Check if it's a file and not a directory
        if os.path.isfile(file_path):
            text = get_pdf_text(file_path)
            df = get_embedding_df(text)
            dfs.append(df)
    return dfs


dfs = pdfs_to_dfs(pdf_file_dir)
for df in dfs:
    print(df)
