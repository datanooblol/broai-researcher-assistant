from pydantic import BaseModel, Field, model_validator
from typing import List, Any, Protocol
from broai.interface import Context

class Payload(BaseModel):
    message:str = Field(description="proxy message updated in each stage and submit along the way")
    original_message:str = Field(description="orginal message added directly by user", default=None)
    potential_jargons:List[str] = Field(description="potential jargons", default=None)
    jargon_contexts:List[Any] = Field(description="jargons and their evidence/explanation for editing a message", default=None)
    jargon_knowledge:str = Field(description="", default=None)
    edited_message:str = Field(description="message edited after getting jargons, later stored at message", default=None)
    sub_queries:List[str] = Field(description="queries broken down from message", default=None)
    context:str = Field(description="proxy contexts", default=None)
    retrieved_contexts:List[Context] = Field(description="contexts retrieved directly from knowledge base, later stored at contexts", default=None)
    reranked_contexts:List[Context] = Field(description="contexts deduplicated, reranked, limit, later stored at contexts", default=None)
    compressed_contexts:List[Context] = Field(description="contexts compressed by extraction or summarization, later stored at contexts", default=None)
    answer:str = Field(description="answer based on message", default=None)

    @model_validator(mode="before")
    def set_original_message(cls, values):
        if values.get("original_message") is None:
            values["original_message"] = values.get("message")
        return values


class BaseAgent(Protocol):
    def run(self, request) -> None: pass


class JargonFlow:
    def __init__(self, jargon_memory, jargon_detector:BaseAgent, jargon_editor:BaseAgent):
        self.jargon_memory = jargon_memory
        self.jargon_detector = jargon_detector
        self.jargon_editor = jargon_editor

    def retrieve_jargons(self, payload):
        jargon_contexts = []
        for jargon in payload.potential_jargons:
            jk = self.jargon_memory.fulltext_search(search_query=jargon)
            if jk is not None:
                jargon_contexts.extend(jk)
        payload.jargon_contexts = jargon_contexts
        jargon_knowledge = "\n\n".join([f"{enum+1}: {j.jargon}\nEvidence: {j.evidence}\nExplanation: {j.explanation}" for enum, j in enumerate(jargon_contexts)])
        payload.jargon_knowledge = jargon_knowledge
    
    def run(self, payload):
        self.jargon_detector.run(payload)
        
        if len(payload.potential_jargons) == 0:
            return None
        self.retrieve_jargons(payload)
        self.jargon_editor.run(payload)

class KnowledgeFlow:
    def __init__(self, longterm_memory, memory_limit=10, search_method:str="vector", reranker=None):
        self.longterm_memory = longterm_memory
        self.memory_limit = memory_limit
        self.reranker = reranker
        self.search_method = search_method

    def rerank(self, search_query, contexts, top_n=10):
        if self.reranker is None:
            return None
        reranked_contexts, scores = self.reranker.run(search_query=search_query, contexts=contexts, top_n=top_n)
        return reranked_contexts

    def deduplicate_contexts(self, contexts):
        deduplicated_contexts = []
        id_list = []
        for c in contexts:
            if c.id not in id_list:
                id_list.append(c.id)
                deduplicated_contexts.append(c)
        return deduplicated_contexts

    def create_context(self, contexts):
        return "\n\n".join([f"{c.context}" for c in contexts])
    
    def retrieve(self, queries:List[str], payload):
        retreived_contexts = []
        for query in queries:
            rc = self.longterm_memory.search(search_query=query, limit=self.memory_limit, search_method=self.search_method)
            retreived_contexts.extend(rc)
            
        deduplicated_contexts = self.deduplicate_contexts(retreived_contexts)
        payload.retrieved_contexts = deduplicated_contexts
        payload.context = self.create_context(deduplicated_contexts)
        reranked_contexts = self.rerank(payload.message, deduplicated_contexts, top_n=self.memory_limit)
        payload.reranked_contexts = reranked_contexts
        if reranked_contexts is not None:
            payload.context = self.create_context(reranked_contexts)
    
    def run(self, payload):
        if payload.sub_queries is not None and isinstance(payload.sub_queries, list):
            queries = payload.sub_queries
        else:
            queries = [payload.message]
        self.retrieve(queries=queries, payload=payload)

class OnlineFlow:
    def __init__(self, 
                 oracle:BaseAgent,
                 knowledge_flow=None,
                 jargon_flow:JargonFlow=None, 
                 query_decomposer:BaseAgent=None, 
                 context_compressor:BaseAgent=None, 
                ):
        self.knowledge_flow = knowledge_flow
        self.jargon_flow = jargon_flow
        self.query_decomposer = query_decomposer
        self.context_compressor = context_compressor
        self.oracle = oracle

    def initialize_payload(self, message):
        self.payload = Payload(message=message)

    def jargon_w_logic(self, ):
        if self.jargon_flow is None:
            return None
        self.jargon_flow.run(self.payload)

    def query_decompose_w_logic(self, ):
        if self.query_decomposer is None:
            return None
        self.query_decomposer.run(self.payload)
    
    def knowledge_w_logic(self, ):
        if self.knowledge_flow is None:
            return None
        self.knowledge_flow.run(self.payload)
    
    def run(self, message):
        self.initialize_payload(message)
        self.jargon_w_logic()
        self.query_decompose_w_logic()
        self.knowledge_w_logic()
        self.oracle.run(self.payload)