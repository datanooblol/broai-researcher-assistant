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

class InputContext(BaseModel):
    context:str = Field(description="the input context.")

class Summary(BaseModel):
    summary:str = Field(description="a short summary based on the provided context.")

prompt_generator = PromptGenerator(
    persona=Persona(name="Andy", description="The best editor who can summarize anything."),
    instructions=Instructions(
        instructions=[
            "summarize the context in concise manner.",
            "do not make up your summary, it must be based on the context only."
        ],
        cautions=[
            "respond in specified JSON format as in examples."
        ],
    ),
    structured_output=Summary,
    examples=Examples(examples=[
        Example(setting="Academic journal", input=InputContext(context="This is the input context."), output=Summary(summary="your summary"))
    ]),
    fallback=Summary(summary="The information is not proper for summarization.")
)

class ContextEnricher:
    def __init__(self):
        self.agent = BroAgent(prompt_generator=prompt_generator, model=bedrock_model)

    def run(self, context:str)->Summary:
        request = InputContext(context=context)
        return self.agent.run(request)