from broai.prompt_management.core import PromptGenerator
from broai.prompt_management.interface import Persona, Instructions, Examples, Example
from broai.experiments.bro_agent import BroAgent
from pydantic import BaseModel, Field
from typing import Tuple, List, Dict, Any, Optional

# you can use any model sharing the same methods: .run, .SystemMessage, .UserMessage, .AIMessage
from broai.llm_management.ollama import BedrockOllamaChat
bedrock_model = BedrockOllamaChat(
    model_name="us.meta.llama3-2-3b-instruct-v1:0",
    temperature=0,
)

class InputMessage(BaseModel):
    message:str = Field(description="A user's input message that you have to understand and respond according to the task")

class DecomposedQueries(BaseModel):
    sub_queries:List[str] = Field(description="A list of decomposed sub queries from the original message")

prompt_generator = PromptGenerator(
    persona=Persona(name="Williams", description="a helpful assistant that specializes in query decomposition."),
    instructions=Instructions(
        instructions=[
            "given a complex, multi-part user input, your task is to break it down into a list of simpler, smaller, searchable sub-queries",
            "each sub-query should focus on a single, specific intent or piece of information",
            "maintain important context from the original query in each sub-query (e.g., timeframes, conditions, specific entities)",
            "if a part of the input requires multiple facts, split it into multiple sub-queries",
            "keep sub-queries concise, but clear enough to be used independently",
            "if the query is already simple, return it as a single-item list"
        ],
        cautions=[
            "Avoid duplicating information between sub-queries unless necessary",
            "respond in specified JSON format as in examples"
        ]
    ),
    structured_output=DecomposedQueries,
    examples=Examples(examples=[
        Example(
            setting="A journal article about climate change",
            input=InputMessage(message="Summarize recent advancements in solar panel technology and how they impact carbon emissions reduction strategies."),
            output=DecomposedQueries(sub_queries=[
                "Recent advancements in solar panel technology",
                "Impact of solar panel technology on carbon emissions reduction strategies"
            ])
        ),
        Example(
            setting="A blog post about personal finance tips",
            input=InputMessage(message="Find advice on saving for retirement in your 30s and strategies for paying off student loans faster."),
            output=DecomposedQueries(sub_queries=[
                "Advice on saving for retirement in your 30s",
                "Strategies for paying off student loans faster"
            ])
        ),
        Example(
            setting="A casual social encounter where someone is planning a trip",
            input=InputMessage(message="Where can I surf in Portugal in September, and what are some good local seafood dishes to try?"),
            output=DecomposedQueries(sub_queries=[
                "Best surfing locations in Portugal in September",
                "Good local seafood dishes to try in Portugal"
            ])
        ),
    ]),
    # fallback=DecomposedQueries(sub_queries=["error"]),
    fallback=None,
)

class QueryDecomposer:
    def __init__(self, ):
        self.agent = BroAgent(prompt_generator=prompt_generator, model=bedrock_model)

    def run(self, payload):
        request = InputMessage(**payload.model_dump())
        sub_queries = self.agent.run(request)
        payload.sub_queries = sub_queries.sub_queries if sub_queries is not None else sub_queries
        return payload.sub_queries