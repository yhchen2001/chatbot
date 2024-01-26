from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
import pandas as pd
#from dotenv import load_dotenv
import requests

#global settings
#load_dotenv()
KEY = "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"
client = OpenAI(api_key=KEY)

def pdfs_to_dfs(directory):
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
        chunks = get_text_chunks(text, chunk_size=500)
        df = pd.DataFrame({'text': chunks})
        embeddings = [get_embedding(c) for c in chunks]
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


def get_image_description(image_url):
    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What’s in this image? describe it in detail within 100 words"},
                    {
                        "type": "image_url",
                        "image_url": image_url,
                    },
                ],
            }
        ],
        max_tokens=300,
    )

    return response.choices[0].message.content

class Embs_Table:
    embeddings_table = None

    @classmethod
    def pdf_to_embs(cls, file_name):
        text = get_pdf_text(file_name)
        cls.embeddings_table = get_embedding_df(text)
    

    @classmethod
    def pdfs_to_embs(cls, directory):
        dfs = []
        for file_name in os.listdir(directory):
            file_path = os.path.join(directory, file_name)
        
            # Check if it's a file and not a directory
            if os.path.isfile(file_path):
                text = get_pdf_text(file_path)
                df = get_embedding_df(text)
                dfs.append(df)
        return dfs

class PastFQ:
    FQ_database = pd.DataFrame(columns=["query", "embs", "response"])
    similarity_threshold = 0.75

    @classmethod
    def save_query(cls, query, response):
        response = client.embeddings.create(
            input=query,
            model="text-embedding-ada-002"
        )
        embs = response.data[0].embedding
        row_data = {
            'query': query,
            'embs': embs,
            'response': response
        }
        row_df = pd.DataFrame([row_data])
        cls.FQ_database = pd.concat([cls.FQ_database, row_df], ignore_index=True)
        print("F&Q database rows:", len(cls.FQ_database))

    @classmethod
    def search_similar_query(cls, question):
        if len(cls.FQ_database) < 5:
            output = ""
        else:
            response = client.embeddings.create(
                input=question,
                model="text-embedding-ada-002"
            )
            question_embedding = response.data[0].embedding
            similarities = [cosine_similarity([question_embedding], [e])[0][0] for e in cls.FQ_database['embs']]
            sorted_similarities = (sorted(enumerate(similarities), key= lambda x: x[1]))[::-1]
            for idx, val in sorted_similarities[:1]:
                if val >= cls.similarity_threshold:
                    output = cls.FQ_database['response'][idx]
                else:
                    output = ""

        return output
        

class conversationRetrievalChain:
    def __init__(self):
        self.chat_history = ""
        self.count = 0
        self.refine = 3
        self.client = OpenAI(api_key=KEY)
        self.df = Embs_Table.embeddings_table
        
    
    def getChatCompletion(self, q):
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", 
                 "content": "You are a helpful assistant and like to interact with people."},

                {"role": "user", 
                 "content": q},
            ]
        )
        response = completion.choices[0].message.content

        return response
    
    def getChatCompletionImage(self, text, img_url):
        response = client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text},
                        {
                            "type": "image_url",
                            "image_url": img_url,
                        },
                    ],
                }
            ],
            max_tokens=300,
        )

        return response.choices[0].message.content
    
    def condensed_question(self, text_input, img_url):
        new_question = ""
        if text_input:
            new_question += text_input
        if img_url:
            img_description = self.getChatCompletionImage("Describe the image as detailed as possible.", img_url)
            new_question += "The image description related to the question is : " + img_description   
        prompt = f"""Given the following conversation and a follow up question,\
                rephrase the follow up question to be a standalone question.\
                Chat History:{self.chat_history}\
                Follow Up Input: {new_question}\
                Standalone question:"""
        
        if len(self.chat_history) == 0:
            condensed_question = new_question
        else:
            condensed_question = self.getChatCompletion(prompt)
        self.chat_history += f'USER: {new_question}\n'

        return condensed_question

    def get_top_k(self, question, top_k = 3):
        question_embedding = get_embedding(question)
        similarities = [cosine_similarity([question_embedding], [e])[0][0] for e in self.df['embeddings']]
        sorted_similarities = (sorted(enumerate(similarities), key= lambda x: x[1]))[::-1]
        related_chunks = [self.df['text'][idx] for idx, val in sorted_similarities[:top_k]]

        return related_chunks
    
    def refine_chat_history(self):
        print("======================= REFINE ========================")
        print("before refine:", self.chat_history)
        get_ongoing_prompt = f"""You are a bot responsible for maintaining conversation histories in a chat room. 
            During conversations, numerous dialogue threads may emerge over time, encompassing concluded topics. 
            To control the total word count in the chat room, you want to retain only the ongoing dialogue threads. 
            This way, users can continue chatting, and you can effectively manage the word count in the chat room. 
            Your task is to separate the chat room conversation into two groups: ongoing and concluded. 
            Please return the ongoing dialogue group to me.

            <EXAMPLE>
            You are provided with a conversation history. Return the ongoing dialogue group to me.
            Conversation History:
            USER:  How's the weather?
            ASSISTANT: Today is rainy outside.
            USER:  How about the temperature outside?
            ASSISTANT: About 24 degree Celsius.
            USER:  I'm hungry. What's good for breakfast?
            ASSISTANT: Toast with chocolate milk will be a good choice.
            USER:  How to make chocolate milk?
            ASSISTANT: You melt the chocolate and mix it with milk.

            Answer:
            Concluded:
            USER:  How's the weather?
            ASSISTANT: Today is rainy outside.
            USER:  How about the temperature outside?＋
            ASSISTANT: About 24 degree Celsius.
            Ongoing:
            USER: I'm hungry. What's good for breakfast?
            ASSISTANT: Toast with chocolate milk will be a good choice.
            USER:  How to make chocolate milk?
            ASSISTANT: You melt the chocolate and mix it with milk.
            <END OF EXAMPLE>

            Now, you are provided with a conversation history. Return the ongoing dialogue group to me.
            Conversation History:
            {self.chat_history}
            Answer:
            Concluded:
            Ongoing:"""
        response = self.getChatCompletion(get_ongoing_prompt)
        try:
            groups = response.split("Ongoing:")
            if groups[1] == "":
                self.chat_history = groups[0]
            else:
                self.chat_history = groups[1]
        except:
            print("parse error!!!!!")

        print("after refine:", self.chat_history)
        print("======================= REFINE ========================")

    def getAnswer(self, user_input=None, img_url=None):
        condensed_input = self.condensed_question(user_input, img_url)
        similarity_answer = PastFQ.search_similar_query(condensed_input)
        if similarity_answer != "":
            print("USED F&Q")
            answer = similarity_answer
        else:
            related_chunks = self.get_top_k(condensed_input)
            prompt = f"""Use the following pieces of context to answer the users question.\
                If you don't know the answer, just say that you don't know, don't try to make up an answer.\
                Context: {str(related_chunks)}\
                Question: {condensed_input}\
                Helpful Answer:"""
            answer = self.getChatCompletion(prompt)
        self.chat_history += f'ASSISTANT: {answer}\n'
        PastFQ.save_query(condensed_input, answer)
        self.count += 1
        if self.count % self.refine == 0:
            self.refine_chat_history()

        return answer
    
if __name__=='__main__':
    Embs_Table.pdf_to_embs("test.pdf")
    chain1 = conversationRetrievalChain()
    print(Embs_Table.embeddings_table['embeddings'][0])
    # chain2 = conversationRetrievalChain()

    # while 1:
    #     q = input("user1: ")
    #     r = chain1.getAnswer(q)
    #     print("assistant1:",r)
    #     q = input("user2: ")
    #     r = chain2.getAnswer(q)
    #     print("assistant2:",r)