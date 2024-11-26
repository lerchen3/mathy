# In actual submission, will use some open source embedding model
# instead of OpenAI's.
from dotenv import load_dotenv
import os
from openai import OpenAI
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if api_key is None:
    raise ValueError("API key not found. Please set the OPENAI_API_KEY environment variable.")

client = OpenAI(api_key=api_key)

def get_embedding(sentence):
    response = client.embeddings.create(
        input=sentence,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding
