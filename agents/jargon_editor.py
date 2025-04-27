from broai.prompt_management.core import PromptGenerator
from broai.prompt_management.interface import Persona, Instructions, Examples, Example
from broai.experiments.bro_agent import BroAgent
from pydantic import BaseModel, Field
from typing import Tuple, List, Dict, Any, Optional

# you can use any model sharing the same methods: .run, .SystemMessage, .UserMessage, .AIMessage
from broai.llm_management.ollama import BedrockOllamaChat
bedrock_model = BedrockOllamaChat(
    model_name="us.meta.llama3-3-70b-instruct-v1:0",
    temperature=0,
)

class InputEditMessage(BaseModel):
    knowledge:str = Field(description="knowledge of jargons, acronyms, abbreviations and proper names")
    message:str = Field(description="a user's input message")

class EditedMessage(BaseModel):
    edited_message:str = Field(description="an edited message based on knowledge and message")

prompt_generator = PromptGenerator(
    persona=Persona(name="Jim", description="You're the best editor who can use your knowledge to edit any message for better clarification."),
    instructions=Instructions(
        instructions=[
            "read knowledge and message carefully",
            "use knowledge to edit the message for better clarification",
        ],
        cautions=[
            "respond in specified JSON format as in examples"
        ]
    ),
    structured_output=EditedMessage,
    examples=Examples(examples=[
        Example(
            setting="Academic Journal",
            input=InputEditMessage(
                knowledge="1: LLM\n Evidence: LLM (Large Language Model) is the advance topic of most research.\nExplanation: LLM is Large Language Model\n\n2: RAG\n Evidence: RAG, retrieval-aumented generation, allows chatbot to connect with the internal knowledge\nExplanation: RAG stands for retrieval-aumented generation",
                message="What are the reasons for using LLM and RAG together?"
            ),
            output=EditedMessage(edited_message="What are the reasons for using LLM, Large Language Model, and RAG, Retrieval-Augmented Generation, together?")
        ),
    ]),
    fallback=EditedMessage(edited_message="error"),
)

JargonEditor = BroAgent(
    prompt_generator=prompt_generator,
    model=bedrock_model
)