import base64
import requests
import os
from dotenv import load_dotenv
import io
import json
import cv2
import numpy as np
import requests


load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
ocr_api_key = os.getenv('OCR_API_KEY')

def ocr(image_path, lang = 'eng'):
    img = cv2.imread(image_path)
    height, width, _ = img.shape
    roi = img[0: height, 0: width]
    url_api = "https://api.ocr.space/parse/image"
    _, compressedimage = cv2.imencode(".jpg", roi, [1, 90])
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

image_path = "testing.png"
print(ocr(image_path))

def getChatCompletionImage(text, image_path):
    # Function to encode the image
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
        
    # Getting the base64 string
    base64_image = encode_image(image_path)

    headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
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
            {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
            }
        ]
        }
    ],
    "max_tokens": 300
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

    return response.json()['choices'][0]['message']['content']
