#test connections with LLMs to make sure they would work


from llama_index.llms.lmstudio import LMStudio
from llama_index.core.base.llms.types import ChatMessage, MessageRole


llm = LMStudio(
    model_name="TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
    base_url="http://localhost:1234/v1",
    temperature=0.0,
)


response = llm.complete("Hey there, what is 2+2?")
print(str(response))









"""
messages = [
    ChatMessage(
        role=MessageRole.SYSTEM,
        content="You an expert AI assistant. Help User with their queries.",
    ),
    ChatMessage(
        role=MessageRole.USER,
        content="What is the significance of the number 42?",
    ),
]
"""