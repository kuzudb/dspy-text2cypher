import asyncio
import os
from typing import Any

import dspy
from dotenv import load_dotenv
from dspy.adapters.baml_adapter import BAMLAdapter
from pydantic import BaseModel, Field

from utils import KuzuDatabaseManager

load_dotenv()

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

# Using OpenRouter. Switch to another LLM provider as needed
lm = dspy.LM(
    model="openrouter/google/gemini-2.0-flash-001",
    api_base="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)
dspy.configure(lm=lm, adapter=BAMLAdapter())


class Query(BaseModel):
    query: str = Field(description="Valid Cypher query with no newlines")


class Property(BaseModel):
    name: str
    type: str = Field(description="Data type of the property")


class Node(BaseModel):
    label: str
    properties: list[Property] | None


class Edge(BaseModel):
    label: str = Field(description="Relationship label")
    from_: Node = Field(alias="from", description="Source node label")
    to: Node = Field(alias="from", description="Target node label")
    properties: list[Property] | None


class GraphSchema(BaseModel):
    nodes: list[Node]
    edges: list[Edge]


class PruneSchema(dspy.Signature):
    """
    Understand the given labelled property graph schema and the given user question. Your task
    is to return ONLY the subset of the schema (node labels, edge labels and properties) that is
    relevant to the question.
      - The schema is a list of nodes and edges in a property graph.
      - The nodes are the entities in the graph.
      - The edges are the relationships between the nodes.
      - Properties of nodes and edges are their attributes, which helps answer the question.
    """

    question: str = dspy.InputField()
    input_schema: str = dspy.InputField()
    pruned_schema: GraphSchema = dspy.OutputField()


class Text2Cypher(dspy.Signature):
    """
    Translate the question into a valid Cypher query that respects the graph schema.

    <SYNTAX>
    - Relationship directions are VERY important to the success of a query. Here's an example: If
    the relationship `hasCreator` is marked as `from` A `to` B, it means that B created A.
    - Use short, concise alphanumeric strings as names of variable bindings (e.g., `a1`, `r1`, etc.)
    - When comparing string properties, ALWAYS do the following:
      - Lowercase the property values before comparison
      - Use the WHERE clause
      - Use the CONTAINS operator to check for presence of one substring in the other
    - DO NOT use APOC as the database does not support it.
    - For datetime queries, use the TIMESTAMP type, which combines the date and time.
    </SYNTAX>

    <RETURN_RESULTS>
    - If the result is an integer, return it as an integer (not a string).
    - When returning results, return property values rather than the entire node or relationship.
    - Do not attempt to coerce data types to number formats (e.g., integer, float) in your results.
    - NO Cypher keywords should be returned by your query.
    </RETURN_RESULTS>
    """

    question: str = dspy.InputField()
    input_schema: str = dspy.InputField()
    query: Query = dspy.OutputField()


class AnswerQuestion(dspy.Signature):
    """
    - Use the provided question, the generated Cypher query and the context to answer the question.
    - If the context is empty, state that you don't have enough information to answer the question.
    """

    question: str = dspy.InputField()
    cypher_query: str = dspy.InputField()
    context: str = dspy.InputField()
    response: str = dspy.OutputField()


class GraphRAG(dspy.Module):
    """
    DSPy custom module that applies Text2Cypher to generate a query and run it
    on the Kuzu database, to generate a natural language response.
    """

    def __init__(self):
        self.prune = dspy.Predict(PruneSchema)
        self.text2cypher = dspy.ChainOfThought(Text2Cypher)
        self.generate_answer = dspy.Predict(AnswerQuestion)

    def get_cypher_query(self, question: str, input_schema: str) -> Query:
        prune_result = self.prune(question=question, input_schema=input_schema)
        schema = prune_result.pruned_schema
        text2cypher_result = self.text2cypher(question=question, input_schema=schema)
        cypher_query = text2cypher_result.query
        return cypher_query

    def run_query(
        self, db_manager: KuzuDatabaseManager, question: str, input_schema: str
    ) -> tuple[str, list[Any] | None]:
        """
        Run a query synchronously on the database.
        """
        result = self.get_cypher_query(question=question, input_schema=input_schema)
        query = result.query
        try:
            # Run the query on the database
            result = db_manager.conn.execute(query)
            results = [item for row in result for item in row]
        except RuntimeError as e:
            print(f"Error running query: {e}")
            results = None
        return query, results

    def forward(self, db_manager: KuzuDatabaseManager, question: str, input_schema: str):
        final_query, final_context = self.run_query(db_manager, question, input_schema)
        if final_context is None:
            print("Empty results obtained from the graph database. Please retry with a different question.")
            return None
        else:
            response = self.generate_answer(
                question=question, cypher_query=final_query, context=str(final_context)
            )
            return response

    async def aforward(self, db_manager: KuzuDatabaseManager, question: str, input_schema: str):
        final_query, final_context = self.run_query(db_manager, question, input_schema)
        if final_context is None:
            print("Empty results obtained from the graph database. Please retry with a different question.")
            return None
        else:
            response = self.generate_answer(
                question=question, cypher_query=final_query, context=str(final_context)
            )
            return response


async def main(questions: list[str]) -> None:
    DB_NAME = "ldbc_1.kuzu"
    db_manager = KuzuDatabaseManager(DB_NAME)
    schema = str(db_manager.get_schema_dict)

    rag = GraphRAG()
    # Run pipeline
    tasks = [
        rag.aforward(db_manager=db_manager, question=question, input_schema=schema)
        for question in questions
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    # print(dspy.inspect_history(n=2))
    print(results)


if __name__ == "__main__":
    questions = [
        "Is there a person with the last name 'Gurung' who's a moderator of a forum with the tag 'Norah_Jones'?",
        "What are the first/last names of people who live in 'Glasgow', and are interested in the tag 'Napoleon'?",
    ]
    asyncio.run(main(questions))
