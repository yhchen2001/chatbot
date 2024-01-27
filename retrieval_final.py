from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
import os
import base64
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
import pandas as pd
from dotenv import load_dotenv
import requests
import io
import json
import cv2
import numpy as np
from pinecone_index import PineconeIndex
from pdf_processing import pdf_to_df
from pinecone import Pinecone, PodSpec
from pymonkey import PyMonkey

#global settings
load_dotenv()
ocr_api_key = os.getenv('OCR_API_KEY')
openai_api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI()

# dfs = pdfs_to_dfs('pdfs')
index_name = 'tsid-grp2'
pi = PineconeIndex(Pinecone(os.getenv('PINECONE_API_KEY')), index_name)
pm = PyMonkey()
# for df in dfs:
#     pi.upsert_pdf(df=df)



class PastFQ:
    FQ_database = pd.DataFrame(columns=["query", "embs", "response"])
    similarity_threshold = 0.95

    @classmethod
    def save_query(cls, query, text_response):
        response = client.embeddings.create(
            input=query,
            model="text-embedding-ada-002"
        )
        embs = response.data[0].embedding
        row_data = {
            'query': query,
            'embs': embs,
            'response': text_response
        }
        row_df = pd.DataFrame([row_data])
        cls.FQ_database = pd.concat([cls.FQ_database, row_df], ignore_index=True)
        #print("F&Q database rows:", len(cls.FQ_database))

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
        self.refine = 6
        self.client = OpenAI()
    
    def getChatCompletion(self, q):
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo-16k",
            messages=[
                {"role": "system", 
                 "content": "You are a helpful assistant and like to interact with people. \
                            Your response should be less than 150 words and in the same language as user input."},

                {"role": "user", 
                 "content": q},
            ],
        )
        response = completion.choices[0].message.content

        return response

    def get_embedding(self, c):
        response = client.embeddings.create(
            input=c,
            model="text-embedding-ada-002"
        )
        return response.data[0].embedding
    
    def getOCR(self, image_paths, lang = 'cht'):
        def get_ocr_text(image_path):
            img = cv2.imread(image_path)
            height, width, _ = img.shape
            roi = img[0: height, 0: width]
            url_api = "https://api.ocr.space/parse/image"
            _, compressedimage = cv2.imencode(".png", roi, [1, 90])
            file_bytes = io.BytesIO(compressedimage)

            result = requests.post(url_api,
                        files = {image_path: file_bytes},
                        data = {"apikey": ocr_api_key,
                                "language": lang})
            result = result.content.decode()
            result = json.loads(result)
            parsed_results = result.get("ParsedResults")[0]
            text_detected = parsed_results.get("ParsedText")

            return text_detected
        
        ocrTexts = ""
        for index, img_path in enumerate(image_paths):
            text = get_ocr_text(img_path)
            ocrTexts += f"img{index+1} ORC text : {text}\n"

        return ocrTexts


    def getChatCompletionImage(self, text, image_paths):
        print("gpt4@@@@@@@@@@")
        def encode_image(image_path):
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
            
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {openai_api_key}"
        }

        payload = {
            "model": "gpt-4-vision-preview",
            "messages": [
                {
                "role": "user",
                "content": [
                    {
                    "type": "text",
                    "text": text
                    },
                ]
                }
            ],
            "max_tokens": 300
        }
        for image_path in image_paths:
            payload["messages"][0]["content"].append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encode_image(image_path)}"
                    }
                }
            )
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

        return response.json()['choices'][0]['message']['content']
        

    def condensed_question(self, text_input, img_paths):
        new_question = ""
        if text_input:
            new_question += text_input
        if len(img_paths) != 0:
            ocr_text = self.getOCR(img_paths)
            description_prompt = f"You are given several images and their corresponding OCR texts. \
                                OCR text : {ocr_text} \
                                Describe the images as detailed as possible. Then summarize the image descriptions and OCR texts into a summary.\
                                Your answer starts with : The images show..."
            img_description = self.getChatCompletionImage(description_prompt, img_paths)
            new_question += "Additional description related to the question is : " + img_description

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
        #print(f"condensed_question : \n {condensed_question} \n")

        return condensed_question

    def get_top_k(self, question, top_k = 5):
        global pi
        print(question)
        question_embedding = self.get_embedding(question)
        related_chunks = pi.query(question_embedding, 1)

        return related_chunks
    
    def refine_chat_history(self):
        print("======================= REFINE ========================")
        get_ongoing_prompt = f"""I will give you a conversation history. You need to split the conversation history into two parts based on semantic meanings.
            Your answer format will be 
            Answer : 
            firstpart: 
            secondpart:

            <EXAMPLE>
            Conversation History:
            USER:  How's the weather?
            ASSISTANT: Today is rainy outside.
            USER:  How about the temperature outside?
            ASSISTANT: About 24 degree Celsius.
            USER:  I'm hungry. What's good for breakfast?
            ASSISTANT: Toast with chocolate milk will be a good choice.
            USER:  How to make chocolate milk?
            ASSISTANT: You melt the chocolate and mix it with milk.

            ANS:
            firstpart:
            USER:  How's the weather?
            ASSISTANT: Today is rainy outside.
            USER:  How about the temperature outside?
            ASSISTANT: About 24 degree Celsius.
            secondpart:
            USER: I'm hungry. What's good for breakfast?
            ASSISTANT: Toast with chocolate milk will be a good choice.
            USER:  How to make chocolate milk?
            ASSISTANT: You melt the chocolate and mix it with milk.
            <END OF EXAMPLE>

            Now, you are provided with a new conversation history. 
            Conversation History:
            {self.chat_history}
            ANS:
            firstpart:
            secondpart:"""
        get_ongoing_prompssst = f"""You are a bot responsible for maintaining conversation histories in a chat room. 
            During conversations, numerous dialogue threads may emerge over time, encompassing concluded topics. 
            To control the total word count in the chat room, you want to retain only the ongoing dialogue threads. 
            This way, users can continue chatting, and you can effectively manage the word count in the chat room. 
            Your task is to separate the chat room conversation into two consecutive parts: ongoing and concluded. 
            Please return the concluded dialogue group and the ongoing dialogue group to me.
            Your answer must include two part in format:
            ANS:
            Concluded:(the concluded dialogues)
            Ongoing:(the ongoing dialogues)
            If you think the whole conversation history is still ongoing, just leave blank in the Concluded part.

            <EXAMPLE>
            Conversation History:
            USER:  How's the weather?
            ASSISTANT: Today is rainy outside.
            USER:  How about the temperature outside?
            ASSISTANT: About 24 degree Celsius.
            USER:  I'm hungry. What's good for breakfast?
            ASSISTANT: Toast with chocolate milk will be a good choice.
            USER:  How to make chocolate milk?
            ASSISTANT: You melt the chocolate and mix it with milk.

            ANS:
            Concluded:
            USER:  How's the weather?
            ASSISTANT: Today is rainy outside.
            USER:  How about the temperature outside?
            ASSISTANT: About 24 degree Celsius.
            Ongoing:
            USER: I'm hungry. What's good for breakfast?
            ASSISTANT: Toast with chocolate milk will be a good choice.
            USER:  How to make chocolate milk?
            ASSISTANT: You melt the chocolate and mix it with milk.
            <END OF EXAMPLE>

            Now, you are provided with a new conversation history. Return the ongoing dialogue group to me.
            Conversation History:
            {self.chat_history}
            ANS:
            Concluded:(the concluded dialogues)
            Ongoing:(the ongoing dialogues)"""
        response = self.getChatCompletion(get_ongoing_prompt)
        #print(response,end="\n")
        try:
            groups = response.split("secondpart:")
            if groups[1] == "":
                self.chat_history = groups[0]
            else:
                self.chat_history = groups[1]
        except:
            textt = self.chat_history
            textt_chunk = textt.split("<>")
            textt_chunk = textt_chunk[3:-1]
            new = ""
            for t in textt_chunk:
                new += f'{t}<>'
            #new = new.rstrip("<>")
            self.chat_history = new
            #print("parse error!!!!!")

        #print("after refine:", self.chat_history)
        #print("======================= REFINE ========================")

    def getAnswer(self, user_input=None, img_paths=None):
        condensed_input = self.condensed_question(user_input, img_paths)
        similarity_answer = PastFQ.search_similar_query(condensed_input)
        if similarity_answer != "":
            #print("!!!!!!!!!!!!!!!!!! USED F&Q !!!!!!!!!!!!!!!!!!!!!!!!!!!")
            answer = similarity_answer
        else:
            related_chunks = self.get_top_k(condensed_input)
            #print("related_chunks",related_chunks)
            
            if len(img_paths) != 0:
                prompt = f"""Use the following pieces of context and given images to answer the users question.\
                    If you don't know the answer, just say that you don't know, don't try to make up an answer.\
                    Context: {str(related_chunks)}\
                    Question: {condensed_input}\
                    Helpful Answer:"""
                answer = self.getChatCompletionImage(prompt, img_paths)
            else:
                prompt = f"""Use the following pieces of context to answer the users question.\
                    If you don't know the answer, just say that you don't know, don't try to make up an answer.\
                    Context: {str(related_chunks)}\
                    Question: {condensed_input}\
                    Helpful Answer:"""
                answer = self.getChatCompletion(prompt)

        self.chat_history += f'ASSISTANT: {answer}<>'
        PastFQ.save_query(condensed_input, answer)
        self.count += 1
        if self.count % self.refine == 0:
            self.refine_chat_history()

        return answer
    
if __name__=='__main__':
    chain1 = conversationRetrievalChain()

    while 1:
        q = input("user1: ")
        r = chain1.getAnswer(user_input=q, img_paths=[])
        print("assistant1:",r)