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

class InputMessage(BaseModel):
    context:str = Field(description="A context served as a knowledge")
    message:str = Field(description="A user's input message that you have to understand and respond according to the task")

class StructuredOutput(BaseModel):
    answer:str = Field(description="An agent's respond to the message based on the task provided in the instructions")

prompt_generator = PromptGenerator(
    persona=Persona(name="John", description="a helpful assistant"),
    instructions=Instructions(
        instructions=[
            "read the message carefully to understand what you have to do to answer the questions",
            "answer questions only based on the information provided in the context",
            "summarize, explain, or rephrase the context if needed to make the answer clear and complete",
            "maintain an honest and helpful tone",
            "if multiple pieces of evidence are found, synthesize them carefully"
        ],
        cautions=[
            "do not invent, assume, or fabricate any facts outside the context",
            "if the information is incomplete or missing, clearly say something like 'Based on the information available, a complete answer is not possible'. do not speculate",
        ]
    ),
)

class WebChatAgent:
    def __init__(self):
        self.agent = BroAgent(prompt_generator=prompt_generator, model=bedrock_model)

    def run(self, payload: BaseModel):
        """
        A wrapper method that extracts and updates the payload, runs the model,
        and populates the answer field in the payload.

        Args:
            payload (BaseModel): A Pydantic BaseModel containing elements used in InputMessage.

        Returns:
            str: The generated answer based on the input message.

        Example:
            ```python
            from pydantic import BaseModel, Field

            class Payload(BaseModel):
                message: str = Field(description="A message used with InputMessage")
                answer: str = Field(description="The answer based on the input message, a default as None indicates that this element will be added later in the process or later stage", default=None)

            payload = Payload(message="Hello There!")
            agent = Agent()
            agent.run(payload)
            print(payload.answer)  # Hi there, nice to meet you!
            ```
        """
        request = InputMessage(**payload.model_dump())
        payload.answer = self.agent.run(request)
        return payload.answer
