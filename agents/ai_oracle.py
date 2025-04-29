from broai.prompt_management.core import PromptGenerator
from broai.prompt_management.interface import Persona, Instructions, Examples, Example
from broai.experiments.bro_agent import BroAgent
from pydantic import BaseModel, Field
from typing import Tuple, List, Dict, Any, Optional

# you can use any model sharing the same methods: .run, .SystemMessage, .UserMessage, .AIMessage
from broai.llm_management.ollama import BedrockOllamaChat
bedrock_model = BedrockOllamaChat(
    model_name="us.meta.llama3-2-11b-instruct-v1:0",
    temperature=0,
)

class InputAIOracle(BaseModel):
    message:str = Field(description="A user's input message that you have to understand and respond according to the task")

prompt_generator = PromptGenerator(
    persona=Persona(name="Betty", description="a wise and trustworthy assistant"),
    instructions=Instructions(
        instructions=[
            "read message carefully and think logically how to answer the message",
            "answer questions based on your knowledge",
            "maintain an honest and helpful tone",
        ],
        cautions=[
            "if you don't know or not sure, do not answer"
        ]
    ),
    fallback="There's something wrong in the system, could you try it again?"
)

class AIOracle:
    def __init__(self):
        self.agent = BroAgent(prompt_generator=prompt_generator,model=bedrock_model)

    def run(self, payload:BaseModel):
        request = InputAIOracle(**payload.model_dump())
        payload.answer = self.agent.run(request)
        return payload.answer