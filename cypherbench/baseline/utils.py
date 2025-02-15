from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_together import ChatTogether
from langchain_fireworks import ChatFireworks
from langchain_google_vertexai import VertexAI
from langchain_community.llms import VLLMOpenAI

LANGCHAIN_LLM_NAME_MAPPING = {
    'llama3.2-3b-fireworks': (ChatFireworks, 'accounts/fireworks/models/llama-v3p2-3b-instruct'),
    'llama3.1-8b-fireworks': (ChatFireworks, 'accounts/fireworks/models/llama-v3p1-8b-instruct'),
    'llama3.1-70b-fireworks': (ChatFireworks, 'accounts/fireworks/models/llama-v3p1-70b-instruct'),
    'llama3.1-405b-fireworks': (ChatFireworks, 'accounts/fireworks/models/llama-v3p1-405b-instruct'),
    'qwen2.5-72b-fireworks': (ChatFireworks, 'accounts/fireworks/models/qwen2p5-72b-instruct'),
    'mixtral-8x22b-fireworks': (ChatFireworks, 'accounts/fireworks/models/mixtral-8x22b-instruct'),
    'mixtral-8x7b-fireworks': (ChatFireworks, 'accounts/fireworks/models/mixtral-8x7b-instruct-hf'),
    'yi-large-fireworks': (ChatFireworks, 'accounts/yi-01-ai/models/yi-large'),

    'gemini1.5-pro-vertex': (VertexAI, 'gemini-1.5-pro-002'),
    'gemini1.5-flash-vertex': (VertexAI, 'gemini-1.5-flash-002'),
    'claude3.5-sonnet-vertex': (VertexAI, 'publishers/anthropic/models/claude-3-5-sonnet'),
    'claude3.5-sonnet-v2-vertex': (VertexAI, 'publishers/anthropic/models/claude-3-5-sonnet-v2'),

    'llama3-8b-groq': (ChatGroq, 'llama3-8b-8192'),
    'llama3-70b-groq': (ChatGroq, 'llama3-70b-8192'),
    'llama3.1-8b-groq': (ChatGroq, 'llama-3.1-8b-instant'),
    'llama3.1-70b-groq': (ChatGroq, 'llama-3.1-70b-versatile'),
    'gemma-7b-groq': (ChatGroq, 'gemma-7b'),
    'gemma2-9b-groq': (ChatGroq, 'gemma2-9b'),

    'llama2-7b-together': (ChatTogether, 'meta-llama/Llama-2-7b-chat-hf'),
    'llama2-13b-together': (ChatTogether, 'meta-llama/Llama-2-13b-chat-hf'),
    'llama2-70b-together': (ChatTogether, 'meta-llama/Llama-2-70b-chat-hf'),
    'llama3-8b-together': (ChatTogether, 'meta-llama/Meta-Llama-3-8B-Instruct-Turbo'),
    'llama3-70b-together': (ChatTogether, 'meta-llama/Meta-Llama-3-70B-Instruct-Turbo'),
    'llama3.1-8b-together': (ChatTogether, 'meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo'),
    'llama3.1-70b-together': (ChatTogether, 'meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo'),

    'llama2-70b-vllm': (VLLMOpenAI, 'meta-llama/Llama-2-70b-chat-hf', 'localhost', 15023, 4096),
    'llama3.1-8b-vllm': (VLLMOpenAI, 'meta-llama/Meta-Llama-3.1-8B-Instruct', 'localhost', 15041, 128000),
    'llama3.1-70b-vllm': (VLLMOpenAI, 'meta-llama/Meta-Llama-3.1-70B-Instruct', 'localhost', 15043, 128000),
}


def get_langchain_llm(llm_name: str, temperature: float):
    if llm_name.startswith('gpt'):
        return ChatOpenAI(model_name=llm_name, temperature=temperature)

    if llm_name not in LANGCHAIN_LLM_NAME_MAPPING:
        raise ValueError(f'Unknown LLM: {llm_name}')

    model_class, model_name = LANGCHAIN_LLM_NAME_MAPPING[llm_name]

    if model_class == VertexAI:
        return model_class(model_name=model_name, temperature=temperature, location='us-east5')

    return model_class(model_name=model_name, temperature=temperature)
