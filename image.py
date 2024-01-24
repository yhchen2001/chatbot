from dotenv import load_dotenv
from retrieval import get_pdf_text, get_embedding_df, upload_to_imgur, conversationRetrievalChain
import os

load_dotenv()
client_id = os.getenv("IMGUR_CLIENT_ID")
from imgur_python import Imgur

def get_image_url(img_path):
    file = os.path.realpath(img_path)
    title = 'Untitled'
    description = 'Image description'
    album = None
    disable_audio = 0

    imgur_client = Imgur({'client_id': client_id})
    response = imgur_client.image_upload(file, title, description, album, disable_audio)
    return response['response']['data']['link']

