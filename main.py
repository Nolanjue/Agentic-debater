from llama_index.core import VectorStoreIndex
from llama_index.readers.obsidian import ObsidianReader
from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()




documents = ObsidianReader("path/to/your/obsidian/vault").load_data()
index = VectorStoreIndex.from_documents(
    documents
)  


"""
# Point to the local server
client = OpenAI(base_url="http://localhost:1234/v1/", api_key=os.environ["OPENAI_API_KEY"])

completion = client.chat.completions.create(
  model="TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
  messages=[
    {"role": "system", "content": "Always answer in rhymes."},
    {"role": "user", "content": "Introduce yourself."}
  ],
  temperature=0.7,
)

print(completion.choices[0].message)"""