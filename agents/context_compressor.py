from broai.prompt_management.core import PromptGenerator
from broai.prompt_management.interface import Persona, Instructions, Examples, Example
from broai.experiments.bro_agent import BroAgent
from pydantic import BaseModel, Field
from typing import Tuple, List, Dict, Any, Optional

# you can use any model sharing the same methods: .run, .SystemMessage, .UserMessage, .AIMessage
from broai.llm_management.ollama import BedrockOllamaChat
bedrock_model = BedrockOllamaChat(
    model_name="us.meta.llama3-3-70b-instruct-v1:0", # 1b, 3b can't pull it off, 11b dead sometimes
    temperature=0,
)

class InputContextCompressor(BaseModel):
    context:str = Field(description="A retrieved context after filtering by reranker")
    query:str = Field(description="A user's input message that you have to understand and respond according to the task")

class ExtractedContext(BaseModel):
    extracted_contexts:List[str] = Field(description="A list of extracted contexts based on given context and query")

prompt_generator = PromptGenerator(
    persona=Persona(name="Kath", description="the best editor who can extract any relevant informations related to the query"),
    instructions=Instructions(
        instructions=[
            "Read the given context and query carefully.",
            "Identify the parts of the context that directly answer or support answering the query.",
            "Extract those parts exactly as they appear in the original context without modifying them.",
            "You may extract multiple separate pieces if necessary to cover the query.",
            "Each extracted piece must be independently meaningful and relevant to the query."
        ],
        cautions=[
            "Do not modify, summarize, rephrase, or invent any part of the context.",
            "Do not extract information that is unrelated or insufficient to answer the query.",
            "Do not combine multiple different sentences if they are not together originally in the context.",
            "Do not extract full paragraphs unless the whole paragraph is necessary to answer the query.",
            "Do not leave an empty list unless absolutely no part of the context can answer the query."
        ]
    ),
    structured_output=ExtractedContext,
    examples=Examples(examples=[
        Example(
            setting="A section from a product manual",
            input=InputContextCompressor(
                context="""
    The Model X500 vacuum cleaner features a high-efficiency particulate air (HEPA) filter that captures 99.97% of dust particles. 
    It also includes a self-cleaning brush roll to reduce maintenance. 
    The vacuum can operate for up to 60 minutes on a single charge, and it comes with a 2-year warranty covering parts and labor.
    """,
                query="How long can the Model X500 vacuum operate on a single charge?"
            ),
            output=ExtractedContext(
                extracted_contexts=[
                    "The vacuum can operate for up to 60 minutes on a single charge."
                ]
            )
        ),
        Example(
            setting="A passage from a health article",
            input=InputContextCompressor(
                context="""
    Regular exercise strengthens the heart and improves circulation. 
    It can also help lower cholesterol and blood pressure levels. 
    However, overexercising without adequate rest can lead to injuries and fatigue.
    """,
                query="What are the benefits of regular exercise?"
            ),
            output=ExtractedContext(
                extracted_contexts=[
                    "Regular exercise strengthens the heart and improves circulation.",
                    "It can also help lower cholesterol and blood pressure levels."
                ]
            )
        ),
    Example(
        setting="A paragraph from a gardening book",
        input=InputContextCompressor(
            context="""
    Tomatoes thrive in warm weather and need at least six hours of sunlight per day. 
    They grow best in well-drained soil with plenty of organic matter. 
    Frequent watering during dry periods is essential to prevent fruit cracking.
    """,
            query="How do you repair a broken laptop screen?"
        ),
        output=ExtractedContext(
            extracted_contexts=["error"]
        )
    ),
    ]),

    fallback=ExtractedContext(extracted_contexts=["error"])
)

ContextCompressor = BroAgent(
    prompt_generator=prompt_generator,
    model=bedrock_model
)