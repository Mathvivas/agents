from dotenv import load_dotenv
import os

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()

# API KEYS
HUGGING_FACE_API_KEY = os.environ.get('HF_TOKEN')
LANGSMITH_API_KEY = os.environ.get('LANGSMITH_API_KEY')


# MODEL
model = HuggingFaceEndpoint(
    repo_id='deepseek-ai/DeepSeek-V3.2',
    task='text-generation',
    temperature=0,
    huggingfacehub_api_token=HUGGING_FACE_API_KEY
)

llm = ChatHuggingFace(
    llm=model
)
