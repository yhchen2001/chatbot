import discord 
import asyncio
from discord import app_commands
from discord.ext import commands

from dotenv import dotenv_values
import os
from retrieval_final import (
    conversationRetrievalChain, 
    pi,
    pm
)
import time
from pdf_processing import pdf_to_df
from utils import get_embeddings, get_text_chunks

config = dotenv_values(".env")
BOT_TOKEN = config['BOT_TOKEN']
GUILD_ID = config['GUILD_ID']
COOL_DOWN = 5

intents = discord.Intents.default()
client = discord.Client(intents=intents)
tree = app_commands.CommandTree(client)

@tree.command(
    name="sayhello",
    description="Say hello to user",
    guild=discord.Object(id=GUILD_ID)
)
async def sayhello(interaction: discord.Interaction, member:discord.Member):
    await interaction.response.send_message(f"咖啡是一種豆漿...")
    # await interaction.followup.defer(ephemeral=True)
    await asyncio.sleep(5)
    await interaction.edit_original_response(content=f"Hello {member.mention}")

@tree.command(
    name="help",
    description="Show guides for using the bot",
    guild=discord.Object(id=GUILD_ID)
)
async def help(interaction: discord.Interaction):
    with open("./help.md", "r") as f:
        lines = f.read().splitlines()
    
    content = '\n'.join(lines)
    embed=discord.Embed(title="Manpage", description=content,color = 0xF1C40F)
    await interaction.response.send_message(embed=embed)
    
pdf_tmp_dir = "tmp_pdf"
try:
    os.mkdir(pdf_tmp_dir)
except:
    pass

@tree.command(
    name="add_pdf",
    description="add pdf",
    guild=discord.Object(id=GUILD_ID)
)
async def addPdf(interaction: discord.Interaction, file:discord.Attachment):
    await interaction.response.send_message(f'You have uploaded: {file.url}')
    print(file.content_type)

    if(not file.filename.endswith('.pdf') and not file.content_type == 'application/pdf'):
        await interaction.followup.send(f"Please upload .pdf file to update model")
        return
    else:
        file_path = f'./{pdf_tmp_dir}/{file.filename}'
        await file.save(f'./{pdf_tmp_dir}/{file.filename}')
    
    if not pm.contain(file.filename):
        df = pdf_to_df(file_path)
        pi.upsert_pdf(df)
        pm.insert([{'name': file.filename, 'content': str(df['text'])}])
    else:
        await interaction.followup.send(f"file has already existed")
    

tmp_dir = "tmp_image"
try:
    os.mkdir(tmp_dir)
except:
    pass

user2chain = {}
user2time = {}

@tree.command(
    name="ask",
    description="ask any questions about the existing files",
    guild=discord.Object(id=GUILD_ID)
)
@app_commands.describe(text='Type your question here', attachment='Attach a file if needed')
async def ask(interaction: discord.Interaction, text: str, attachment: discord.Attachment = None):
    print("hiiiiii=======")
    user_id = interaction.user.id
    if user_id not in user2chain:
        user2chain[user_id] = conversationRetrievalChain()
        user2time[user_id] = time.time()
    elif time.time() - user2time[user_id] < COOL_DOWN:
        await interaction.response.send_message(f'You have query too frequently')
        return
    chain = user2chain[user_id]
    file_names = []
    if attachment and any(attachment.filename.lower().endswith(ext) for ext in ['png', 'jpg', 'jpeg']):
        # save all file as .jpg
        image_path = f'./{tmp_dir}/{attachment.filename}'
        await attachment.save(image_path)

        name, ext = os.path.splitext(image_path)
        new_image_path = f'./{name}.jpg'
        os.rename(image_path, new_image_path)
        file_names.append(new_image_path)
    elif attachment:
        await interaction.response.send_message(f'not a image')

    await interaction.response.send_message('starts answering')

    reply = chain.getAnswer(text, file_names)
    print(reply)

    # remove tmp files
    for file_name in file_names:
        os.remove(file_name)

    chunks = get_text_chunks(reply)
    for chunk in chunks:
        await  interaction.followup.send(chunk)
    

    
    # await interaction.response.send_message(f'You have uploaded: {file.url}')



# sync commands to discord app when the client is ready
@client.event
async def on_ready():
    await tree.sync(guild=discord.Object(id=GUILD_ID))
    print("Sync commands to discord bot successfully")

client.run(BOT_TOKEN)