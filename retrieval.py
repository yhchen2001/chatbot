from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
import pandas as pd
from dotenv import load_dotenv
import requests

top_k = 2
chunk_size = 500

load_dotenv()
client = OpenAI()

def get_embedding(c):
    response = client.embeddings.create(
        input=c,
        model="text-embedding-ada-002"
    )
    return response.data[0].embedding

def get_pdf_text(file_name):
    reader = PdfReader(file_name)
    number_of_pages = len(reader.pages)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def get_text_chunks(text, chunk_size = 1000):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_embedding_df(text):
    chunks = get_text_chunks(text, chunk_size=chunk_size)
    df = pd.DataFrame({'text': chunks})
    print(len(chunks))
    
    embeddings = [get_embedding(c) for c in chunks]
    df['embeddings'] = embeddings

    return df

def get_image_description(image_url):
    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What’s in this image? describe as detailly as possible"},
                    {
                        "type": "image_url",
                        "image_url": image_url,
                    },
                ],
            }
        ],
        max_tokens=300,
    )

    return response.choices[0]

def upload_to_imgur(image_path, client_id):
    """
    Uploads an image to Imgur and returns the URL.
    """
    headers = {'Authorization': f'Client-ID {client_id}'}

    with open(image_path, 'rb') as image:
        data = {'image': image.read()}
        response = requests.post('https://api.imgur.com/3/upload', headers=headers, files=data)
    
    if response.status_code == 200:
        return response.json()['data']['link']
    else:
        return None

# messages = [{"role": "system", "content": "You are a helpful assistant."}]
# def chat(message):
#     messages.append({"role": "user", "content": message})
#     print(messages)

#     completion = client.chat.completions.create(
#         model="gpt-3.5-turbo",
#         messages=messages
#     )

#     response = completion.choices[0].message.content
#     messages.append({'role':'assistant', 'content': response})
#     return response

class conversationRetrievalChain:
    def __init__(self, df) -> None:
        self.user_messages = ""
        self.client = OpenAI()
        self.df = df
    
    def clearMessage(self):
        self.user_messages = ""
    
    def getChatCompletion(self, q):
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": q},
            ]
        )
        response = completion.choices[0].message.content
        return response
    
    def addUserMessage(self, m):
        prompt = f"""Combine the chat history and follow up question into \
                a standalone question. Chat History: {self.user_messages} \
                Follow up question: {m}"""
        new_message = self.getChatCompletion(prompt)
        self.user_messages = new_message

    def get_top_k(self, question, top_k = 1):
        question_embedding = get_embedding(question)

        similarities = [cosine_similarity([question_embedding], [e])[0][0] for e in self.df['embeddings']]
        sorted_similarities = (sorted(enumerate(similarities), key= lambda x: x[1]))[::-1]

        related_chunks = [self.df['text'][idx] for idx, val in sorted_similarities[:top_k]]
        return related_chunks
    
    def getAnswer(self, m):
        self.addUserMessage(m)
        print("new question:", self.user_messages)
        related_chunks = self.get_top_k(m, 1)
        m = "content: " +  str(related_chunks) + "question: " + self.user_messages + "\nplease answer the question according to the content"
        return self.getChatCompletion(m)
        
if __name__=='__main__':
    file_name = "test.pdf"

    text = get_pdf_text(file_name)
    df = get_embedding_df(text)
    chain = conversationRetrievalChain(df)

    while 1:
        q = input("user: ")
        r = chain.getAnswer(q)
        print("assistant:",r)

