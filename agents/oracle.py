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

class InputOracle(BaseModel):
    prior_knowledge:str = Field(description="The prior knowledge that you have")
    message:str = Field(description="A user's input message that you have to understand and respond according to the task")

class Answer(BaseModel):
    answer:str = Field(description="An agent's answer based on the knowledge and message.")

prompt_generator = PromptGenerator(
    persona=Persona(name="Kate", description="a wise and trustworthy assistant"),
    instructions=Instructions(
        instructions=[
            "answer questions only based on the information provided from your prior knowledge",
            "use only the given information from the prior knowledge to form your response",
            "summarize, explain, or rephrase the prior knowledge if needed to make the answer clear and complete",
            "maintain an honest and helpful tone",
            "if multiple pieces of evidence are found, synthesize them carefully"
        ],
        cautions=[
            "do not invent, assume, or fabricate any facts outside prior knowledge",
            "if the information is incomplete or missing, clearly say something like 'Based on the information available, a complete answer is not possible'. do not speculate",
            "respond in specified JSON format as in examples"
        ]
    ),
    structured_output=Answer,
    examples=Examples(examples=[
        Example(
            setting="A journal article about the effects of meditation on mental health",
            input=InputOracle(
                prior_knowledge="Studies have shown that regular meditation can decrease symptoms of anxiety and depression. It can also improve attention span and emotional regulation.",
                message="How does meditation impact mental health?"
            ),
            output=Answer(
                answer="Regular meditation can decrease symptoms of anxiety and depression, and also improve attention span and emotional regulation."
            )
        ),
        Example(
            setting="A blog post about the history of coffee",
            input=InputOracle(
                prior_knowledge="Coffee originated in Ethiopia. It spread to the Arabian Peninsula where it became an important part of cultural life by the 15th century.",
                message="Where did coffee originate and when did it become popular?"
            ),
            output=Answer(
                answer="Coffee originated in Ethiopia and became an important part of cultural life in the Arabian Peninsula by the 15th century."
            )
        ),
    ]),
    fallback=Answer(answer="Could you be more specific?")
)

Oracle = BroAgent(
    prompt_generator=prompt_generator,
    model=bedrock_model
)