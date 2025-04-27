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

class InputContext(BaseModel):
    context:str = Field(description="the input context.")

class Jargon(BaseModel):
    jargon:str = Field(description="the extracted jargon, acronym, abbreviation, proper name from the context"),
    evidence:str = Field(description="the evidence of the extracted jargon, acronym, abbreviation, proper name from the context"),
    explanation:str = Field(description="A short explanation based on the evidence.")

class Jargons(BaseModel):
    jargons:List[Jargon] = Field(description="A list of extracted jargons")

prompt_generator = PromptGenerator(
    persona=Persona(name="Jame", description="The best proof reader who can spot any jargon, acronym, abbreviation and proper name at ease."),
    instructions=Instructions(
        instructions=[
            "extract jargon, acronym, abbrevation, proper name and its evidence.",
            "create a short explanation of the extracted jargon, acronym, abbrevation, perper name based on its evidence.",
            "always avoid extracting the person name",
            "do not make up your explanation",
        ],
        cautions=[
            "Remember not everything is jargon, acronym, abbrevation or proper name, so do not try to extract everything, try only obvious or certain."
            "Do not extract jargon, acronym, abbreviation or proper name that does not contain evidence",
            "if you are not sure if it's a jargon, acronym, abbreviation, proper name, do not extract it",
            "respond in specified JSON format as in examples"
        ]
    ),
    structured_output=Jargons,
    examples=Examples(examples=[
        Example(
            setting="Academic Journal",
            input=InputContext(context="Andrew Ng said, The new era of LLM, large language model, comes faster than anyone expected. It works well with RAG, retrieveal augmented generation, and it allows agentic framework to ..."),
            output=Jargons(jargons=[
                Jargon(
                    jargon="LLM",
                    evidence="The new era of LLM, large language model",
                    explanation="LLM is a large language model"
                ),
                Jargon(
                    jargon="RAG",
                    evidence="It works well with RAG, retrieval augmented generation",
                    explanation="RAG is short for retrieval augmented generation"
                )
            ])
        )
    ]),
    fallback=Jargons(jargons=[Jargon(jargon="error", evidence="error", explanation="error")])
)

JargonExtractor = BroAgent(
    prompt_generator=prompt_generator,
    model=bedrock_model
)