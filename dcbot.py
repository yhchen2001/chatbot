import openai
import discord
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
import pandas as pd
from dotenv import load_dotenv
from retrieval_v2 import (
    get_pdf_text, 
    get_embedding_df, 
    conversationRetrievalChain, 
    get_text_chunks,
    Embs_Table,
    get_image_description
)
import asyncio

load_dotenv()
token = os.getenv("DISCORD_TOKEN")
url = 'https://discord.com/api/oauth2/authorize?client_id=1126795472061861928&permissions=8&scope=bot'

intents = discord.Intents.all()
client = discord.Client(intents=intents)

pdf_dir = 'pdfs'
file_name = "test.pdf"
Embs_Table.pdf_to_embs("test.pdf")

tmp_dir = "tmp_image"

try:
    os.mkdir(tmp_dir)
except:
    pass


@client.event
async def on_ready():
    print(f"We have logged in as {client.user}")

users = {}
# Event triggered when a message is received
@client.event
async def on_message(message: discord.Message):
    # Check if the message was sent by the bot itself to avoid an infinite loop
    if message.author == client.user:
        return

    if message.author.id not in users:
        chain = conversationRetrievalChain()
        users[message.author.id] = chain
    else:
        chain = users[message.author.id]
    
    text = ""
    if message.clean_content:
        text = message.clean_content
    
    file_paths = []
    if message.attachments:
        for attachment in message.attachments:
            if any(attachment.filename.lower().endswith(ext) for ext in ['png', 'jpg', 'jpeg', 'JPEG', 'JPG']):
                # save all file as .jpg
                image_name = f'./{attachment.filename}'
                await attachment.save(f'./{tmp_dir}/{image_name}')
                name, ext = os.path.splitext(image_name)
                new_image_name = name + ".jpg"
                os.rename(f'./{tmp_dir}/{image_name}', f'./{tmp_dir}/{new_image_name}')

    reply = chain.getAnswer(text, file_paths)

    # remove all saved image
    for filename in os.listdir(tmp_dir):
        file_path = os.path.join(tmp_dir, filename)

        # Check if it's a file and not a directory
        if os.path.isfile(file_path):
            os.remove(file_path)
    print(reply)

    reply = reply.split("```")

    reply[1:] = ["```" + chunk + "```" if chunk[0] != '\n' else chunk for chunk in reply[1:]]

    for chunk in reply:
        smaller_chunks = get_text_chunks(chunk)
        for smaller_chunk in smaller_chunks:
            await message.channel.send(smaller_chunk)


# Run the bot with your Discord bot token
client.run(token)
