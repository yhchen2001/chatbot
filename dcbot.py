import openai
import discord
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
import pandas as pd
from dotenv import load_dotenv
from retrieval import get_pdf_text, get_embedding_df, upload_to_imgur, conversationRetrievalChain

load_dotenv()
file_name = "test.pdf"

text = get_pdf_text(file_name)
df = get_embedding_df(text)
print(df)
chain = conversationRetrievalChain(df)

# while 1:
#     q = input("user: ")
#     r = chain.getAnswer(q)
#     print("assistant:",r)

token = 'MTEyNjc5NTQ3MjA2MTg2MTkyOA.GzBuJw.FAdqWt0wbP4TSWqqgPErpBztgj18dzfq3OplgM'
url = 'https://discord.com/api/oauth2/authorize?client_id=1126795472061861928&permissions=8&scope=bot'

chat_log = []

intents = discord.Intents.all()
client = discord.Client(intents=intents)

@client.event
async def on_ready():
    print(f"We have logged in as {client.user}")

# Event triggered when a message is received
@client.event
async def on_message(message: discord.Message):
    # Check if the message was sent by the bot itself to avoid an infinite loop
    if message.author == client.user:
        return
    text = message.clean_content
    reply = chain.getAnswer(text)
    print(reply)

    reply = reply.split("```")

    reply[1:] = ["```" + chunk + "```" if chunk[0] != '\n' else chunk for chunk in reply[1:]]

    for chunk in reply:
        await message.channel.send(chunk)
    
    if message.attachments:
        for attachment in message.attachments:
            if any(attachment.filename.lower().endswith(ext) for ext in ['png', 'jpg', 'jpeg', 'gif']):
                image_path = f'./{attachment.filename}'
                await attachment.save(image_path)

                # Upload to Imgur
                imgur_url = upload_to_imgur(image_path, 'YOUR_IMGUR_CLIENT_ID')

                if imgur_url:
                    await message.channel.send(f"Image uploaded to Imgur: {imgur_url}")
                else:
                    await message.channel.send("Failed to upload image to Imgur.")


# Run the bot with your Discord bot token
client.run(token)
