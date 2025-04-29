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

class InputKBOracle(BaseModel):
    context:str = Field(description="The prior knowledge based on the context that you have")
    message:str = Field(description="A user's input message that you have to understand and respond according to the task")

prompt_generator = PromptGenerator(
    persona=Persona(name="Kate", description="a wise and trustworthy assistant"),
    instructions=Instructions(
        instructions=[
            "read the message carefully to understand what you have to do to answer the questions",
            "answer questions only based on the information provided in the context",
            # "use only the given information from the context to form your response",
            "summarize, explain, or rephrase the context if needed to make the answer clear and complete",
            "maintain an honest and helpful tone",
            "if multiple pieces of evidence are found, synthesize them carefully"
        ],
        cautions=[
            "do not invent, assume, or fabricate any facts outside the context",
            "if the information is incomplete or missing, clearly say something like 'Based on the information available, a complete answer is not possible'. do not speculate",
        ]
    ),
    # structured_output=Answer,
    examples=Examples(examples=[
        Example(
            setting="A journal article about the effects of meditation on mental health",
            input=InputKBOracle(
                context="Studies have shown that regular meditation can decrease symptoms of anxiety and depression. It can also improve attention span and emotional regulation.",
                message="How does meditation impact mental health?"
            ),
            output="Regular meditation can decrease symptoms of anxiety and depression, and also improve attention span and emotional regulation."
        ),
        Example(
            setting="A blog post about the history of coffee",
            input=InputKBOracle(
                context="Coffee originated in Ethiopia. It spread to the Arabian Peninsula where it became an important part of cultural life by the 15th century.",
                message="Where did coffee originate and when did it become popular?"
            ),
            output="Coffee originated in Ethiopia and became an important part of cultural life in the Arabian Peninsula by the 15th century.",
            
        ),
    ]),
)

KBOracle = BroAgent(
    prompt_generator=prompt_generator,
    model=bedrock_model
)

class KBOracle:
    def __init__(self, ):
        self.agent = BroAgent(prompt_generator=prompt_generator, model=bedrock_model)

    def run(self, payload):
        request = InputKBOracle(**payload.model_dump())
        payload.answer = self.agent.run(request)
        return payload.answer
        