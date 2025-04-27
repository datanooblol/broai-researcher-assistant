import duckdb
from broai.experiments.huggingface_embedding import BaseEmbeddingModel
from typing import Dict, List, Any
from broai.duckdb_management.utils import get_create_table_query, get_insert_query

class JargonStore:
    def __init__(self, db_name:str, table:str, embedding:BaseEmbeddingModel, limit:int=5):
        self.db_name = db_name
        self.table = table
        self.embedding_model = validate_baseclass(embedding, "embedding", BaseEmbeddingModel)
        self.embedding_size = self.embedding_model.run(["test"]).shape[1]
        self.limit = limit
        self.__schemas = {
            "id": "VARCHAR",
            "jargon": "VARCHAR",
            "evidence": "VARCHAR",
            "explanation": "VARCHAR",
            "metadata": "JSON",
        }
        self.create_table()
    
    def sql(self, query, params:Dict[str, Any]=None):
        with duckdb.connect(self.db_name) as con:
            con.sql(query, params=params)
            
    def sql_df(self, query, params:Dict[str, Any]=None):
        with duckdb.connect(self.db_name) as con:
            df = con.sql(query, params=params).to_df()
        return df
        
    def create_table(self,):
        query = get_create_table_query(table=self.table, schemas=self.__schemas)
        self.sql(query)

    def add_jargons(self, jargons:list):
        rows = jargons
        with duckdb.connect(self.db_name) as con:
            con.executemany(f"INSERT INTO {self.table} (id, jargon, evidence, explanation, metadata) VALUES (?, ?, ?, ?, ?)", rows)
        self.create_fts_index()

    def create_fts_index(self):
        query = f"""
        INSTALL fts;
        LOAD fts;
        PRAGMA create_fts_index(
            '{self.table}', 'id', 'jargon', overwrite=1
        );
        """.strip()
        self.sql(query)

    def fulltext_search(self, search_query:str, limit:int=5):
        query = f"""\
        SELECT *
        FROM (
            SELECT *, fts_main_{self.table}.match_bm25(
                id,
                '{search_query}',
                fields := 'jargon'
            ) AS score
            FROM {self.table}
        ) sq
        ORDER BY score DESC;
        """
        return self.sql_df(query=query, params=None)