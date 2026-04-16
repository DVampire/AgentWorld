import sys
import os
from dotenv import load_dotenv
load_dotenv(verbose=True)
from pathlib import Path
import asyncio
import json

root = str(Path(__file__).resolve().parents[1])
sys.path.append(root)

from src.message import (
    HumanMessage, 
    SystemMessage, 
    ContentPartText,
    ContentPartImage,
    ImageURL,
    ContentPartAudio,
    AudioURL,
    ContentPartVideo,
    VideoURL,
)
from src.utils import make_file_url

async def test_message():

    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content=[
            ContentPartText(text="What are the names of the Pokémon in the image?"),
            ContentPartImage(image_url=ImageURL(url=make_file_url(file_path="tests/files/pokemon.jpg"))),
            ContentPartAudio(audio_url=AudioURL(url=make_file_url(file_path="tests/files/audio.mp3"))),
            ContentPartVideo(video_url=VideoURL(url=make_file_url(file_path="tests/files/video.MOV"))),
        ]),
    ]
    
    print(messages)
    
if __name__ == "__main__":
    asyncio.run(test_message())