from typing import List, Any, Protocol
from broai.interface import Context
from broai.experiments.chunk import split_markdown, consolidate_markdown, get_markdown_sections, split_overlap
from tqdm import tqdm
from pydantic import BaseModel, Field
from package.jargon_store import JargonRecord

class Payload(BaseModel):
    source:str = Field(description="")
    source_type:str = Field(description="")
    text:str = Field(description="", default=None)
    contexts:List[Context] = Field(description="", default=None)
    enriched_contexts:List[Context] = Field(description="", default=None)
    longterm_contexts:List[Context] = Field(description="", default=None)
    jargon_contexts:List[Context] = Field(description="", default=None)


class BaseAgent(Protocol):
    def run(self, request) -> None: pass

class MemoryFlow:
    def __init__(
        self, 
        raw_memory, 
        enrich_memory, 
        longterm_memory, 
        jargon_memory
    ):
        self.raw_memory = raw_memory
        self.enrich_memory = enrich_memory
        self.longterm_memory = longterm_memory
        self.jargon_memory = jargon_memory

class OfflineFlow:
    def __init__(self,
                 knowledge_flow:MemoryFlow,
                 context_enricher:BaseAgent,
                 jargon_extractor:BaseAgent,
                 max_tokens:int=500,
                 overlap:int=int(500*0.3)
                ):
        self.knowledge_flow = knowledge_flow
        self.context_enricher = context_enricher
        self.jargon_extractor = jargon_extractor
        self.max_tokens = max_tokens
        self.overlap = overlap

    def initialize_payload(self, source, source_type):
        self.payload = Payload(
            source=source, 
            source_type=source_type
        )
    
    def load_and_convert_markdown(self) -> str:
        with open(self.payload.source, "r") as f:
            text = f.read()
        self.payload.text = text

    def chunk(self, ):
        source = self.payload.source
        source_type = self.payload.source_type
        chunks = split_markdown(self.payload.text)
        consolidated_chunks = consolidate_markdown(chunks)
        sections = get_markdown_sections(consolidated_chunks)
        contexts = []
        for section, chunk in zip(sections, consolidated_chunks):
            contexts.append(
                Context(context=chunk, metadata={"section": section, "source": source, "type": source_type})
            )
        new_contexts = split_overlap(contexts, max_tokens=self.max_tokens, overlap=self.overlap)
        self.payload.contexts = new_contexts

    def _enrich(self):
        summaries = []
        errors = []
        for enum, context in enumerate(tqdm(self.payload.contexts)):
            try:
                response = self.context_enricher.run(context=context.context)
                summaries.append(
                    Context(id=context.id, context=response.summary, metadata=context.metadata.copy())
                )
            except Exception as e:
                errors.append((enum, e, context))
        self.payload.enriched_contexts = summaries

    def _combo(self):
        combo_contexts = []
        for context, enriched in zip(self.payload.contexts, self.payload.enriched_contexts):
            if context.id == enriched.id:
                combo_contexts.append(
                    Context(id=context.id, context=f"{enriched.context}\n\n{context.context}", metadata=context.metadata.copy())
                )
        self.payload.longterm_contexts = combo_contexts

    def _jargon(self):
        jargon_contexts = []
        errors = []
        for enum, context in enumerate(tqdm(self.payload.contexts)):
            try:
                response = self.jargon_extractor.run(context=context.context)
                metadata = context.metadata.copy()
                if response is not None:
                    for j in response:
                        if j.evidence.lower() != "none":
                            jr = JargonRecord(jargon=j.jargon, evidence=j.evidence, explanation=j.explanation, metadata=metadata.copy())
                            jargon_contexts.append(jr)
            except Exception as e:
                print(e)
                errors.append((enum, str(e), context))
        self.payload.jargon_contexts = jargon_contexts
        print(len(errors))
    
    def run(self, source:str, source_type:str):
        self.initialize_payload(source, source_type)
        self.load_and_convert_markdown()
        self.chunk()
        self.knowledge_flow.raw_memory.add_contexts(self.payload.contexts)
        self._enrich()
        self.knowledge_flow.enrich_memory.add_contexts(self.payload.enriched_contexts)
        self._combo()
        self.knowledge_flow.longterm_memory.add_contexts(self.payload.longterm_contexts)
        self._jargon()
        self.knowledge_flow.jargon_memory.add_jargons(self.payload.jargon_contexts)