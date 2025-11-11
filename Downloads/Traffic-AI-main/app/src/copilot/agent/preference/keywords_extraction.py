# Standard Library Imports
import logging
from typing import (
    Awaitable,
    Callable,
    Dict,
    Optional,
    Tuple,
    Union,
    List
)
from unittest.mock import Base

# Third-party library imports
from pydantic import ValidationError

# Core Module Imports
from autogen_core import (
    MessageContext,
    TopicId,
    SingleThreadedAgentRuntime,
    message_handler,
)
from autogen_core.models import (
    CreateResult,
    SystemMessage,
    UserMessage,
)

# Project-Specific Imports
from src.copilot.agent.agent import Agent
from src.copilot.chat_client.chat_client import ChatClient
from src.connector.agentops.agentops import LogMessage
from src.connector.agentops.langfuse_ops import LangfuseOps
from src.copilot.agent.preference.constant import (
    KeywordsExtractionRequest,
    KeywordsExtractionResponse,
    KEYWORDS_EXTRACTION_REQUEST,
    KEYWORDS_EXTRACTION_TOPIC
)

import asyncio
import json
import logging
from dataclasses import asdict
from typing import (
    Any, Annotated, Awaitable, Callable, Dict, List, Optional, Set, Tuple
)
from mlflow.entities import SpanType, Document

from pydantic import BaseModel

from autogen_core import (
    FunctionCall,
    MessageContext,
    SingleThreadedAgentRuntime,
    TopicId,
    message_handler,
)
from autogen_core.models import (
    AssistantMessage,
    CreateResult,
    FunctionExecutionResult,
    FunctionExecutionResultMessage,
    SystemMessage,
    UserMessage,
)
from autogen_core.tools import FunctionTool, Tool

from src.connector.agentops.agentops import LogMessage
from src.copilot.agent.agent import Agent
from src.copilot.chat_client.chat_client import ChatClient
from src.copilot.utils.gather import run_tool_with_retries
from src.connector.agentops.langfuse_ops import LangfuseOps

logger = logging.getLogger(__name__)

import pandas as pd
from dataclasses import dataclass
from rapidfuzz import process, fuzz
from collections import defaultdict, deque
import uuid
import asyncio

@dataclass(frozen=True)
class Relation:
    source_id: str
    target_id: str
    relation_type: str

class Graph:
    def __init__(self):
        self.nodes = set()
        self.relations = set()
        self.adjacency = defaultdict(lambda: defaultdict(set))  # adjacency[src][dst] = set(Relation)

    def add_node(self, name: str) -> str:
        self.nodes.add(name)
        return name

    def add_relation(self, relation: Relation):
        self.relations.add(relation)
        self.adjacency[relation.source_id][relation.target_id].add(relation)
        self.adjacency[relation.target_id][relation.source_id].add(relation)  # undirected

    # Removed @lru_cache to avoid coroutine reuse issues
    async def search(
        self, keyword, fuzzy=False, threshold=80, limit=10, max_hops=3, max_per_type=10
    ):
        loop = asyncio.get_event_loop()
        
        keyword = keyword.upper()

        # Fuzzy or exact match for starting nodes
        if fuzzy:
            # Run RapidFuzz in executor to avoid blocking event loop
            matches = await loop.run_in_executor(
                None,
                lambda: process.extract(keyword, self.nodes, scorer=fuzz.WRatio, limit=limit)
            )
            start_nodes_scores = {node: score for node, score, *_ in matches if score >= threshold}
        else:
            start_nodes_scores = {keyword: 100} if keyword in self.nodes else {}

        if not start_nodes_scores:
            return {"nodes": [], "relations": [], "similarity": {}}

        visited_nodes = set(start_nodes_scores)
        traversed_relations = set()
        queue = deque([(node, 0) for node in start_nodes_scores])

        while queue:
            current_node, hop = queue.popleft()
            if hop >= max_hops:
                continue
            neighbors = self.adjacency[current_node]
            # Group outgoing relations by relation_type
            reltype_to_relations = defaultdict(list)
            for neighbor, relations in neighbors.items():
                for r in relations:
                    reltype_to_relations[r.relation_type].append((neighbor, r))
            # Expand neighbors per relation type, limited
            for rel_type, neighbor_rel_list in reltype_to_relations.items():
                if len(neighbor_rel_list) > max_per_type:
                    continue
                for neighbor, r in neighbor_rel_list:
                    traversed_relations.add(r)
                    if neighbor not in visited_nodes:
                        visited_nodes.add(neighbor)
                        queue.append((neighbor, hop + 1))

        relations_serialized = [
            {"source_id": r.source_id, "target_id": r.target_id, "relation_type": r.relation_type}
            for r in traversed_relations
        ]

        return {
            "nodes": list(visited_nodes),
            "relations": relations_serialized,
            "similarity": start_nodes_scores
        }
        
    def merge(self, other: "Graph"):
        self.nodes.update(other.nodes)
        self.relations.update(other.relations)
        for src, dsts in other.adjacency.items():
            for dst, rels in dsts.items():
                self.adjacency[src][dst].update(rels)

    @classmethod
    def build_from_geo_csv(cls, csv_file_path: str) -> "Graph":
        g = cls()
        df = pd.read_csv(csv_file_path)
        # Category nodes
        category_nodes = {
            "PORT": g.add_node("PORT"),
            "UNLOCODE": g.add_node("UNLOCODE"),
            "PORT_CODE": g.add_node("PORT_CODE"),
            "PORT_NAME": g.add_node("PORT_NAME"),
            "PORT_OF_LOADING": g.add_node("PORT_OF_LOADING"),
            "PORT_OF_DISCHARGE": g.add_node("PORT_OF_DISCHARGE"),
            "TRANSHIPMENT_PORT": g.add_node("TRANSHIPMENT_PORT"),
        }
        # Add category relations
        g.add_relation(Relation(category_nodes["PORT"], category_nodes["PORT_OF_LOADING"], 'can_be'))
        g.add_relation(Relation(category_nodes["PORT"], category_nodes["PORT_OF_DISCHARGE"], 'can_be'))
        g.add_relation(Relation(category_nodes["PORT"], category_nodes["TRANSHIPMENT_PORT"], 'can_be'))

        # Iterate rows efficiently
        for row in df.itertuples(index=False):
            geo_loc_uuid = getattr(row, 'geo_loc_uuid', None)
            if pd.notnull(geo_loc_uuid):
                port_node_id = g.add_node(str(geo_loc_uuid))
                g.add_relation(Relation(port_node_id, category_nodes["PORT"], 'is_a'))

                # Alias/code relations
                fields = [
                    ('name', 'has_alias', None),
                    ('unlocode', 'has_code', "UNLOCODE"),
                    ('internal_cde', 'has_code', "PORT_CODE"),
                    ('default_nme', 'has_alias', "PORT_NAME"),
                ]
                for field, rel_type, cat_key in fields:
                    value = getattr(row, field, None)
                    if pd.notnull(value):
                        node_id = g.add_node(str(value).upper())
                        g.add_relation(Relation(port_node_id, node_id, rel_type))
                        if cat_key:
                            g.add_relation(Relation(node_id, category_nodes[cat_key], 'is_a'))
        return g

    @classmethod
    def build_from_svc_csv(cls, csv_file_path: str) -> "Graph":
        g = cls()
        df = pd.read_csv(csv_file_path)
        # Category nodes
        category_nodes = {
            "SERVICE": g.add_node("SERVICE"),
            "SERVICE_CODE": g.add_node("SERVICE_CODE"),
            "SERVICE_NAME": g.add_node("SERVICE_NAME"),
        }

        # Iterate rows efficiently
        for row in df.itertuples(index=False):
                service_node_id = g.add_node(str(uuid.uuid4()))
                g.add_relation(Relation(service_node_id, category_nodes["SERVICE"], 'is_a'))

                # Alias/code relations
                fields = [
                    ('svc_cde', 'has_code', "SERVICE_CODE"),
                    ('svc_nme', 'has_alias', "SERVICE_NAME"),
                ]
                for field, rel_type, cat_key in fields:
                    value = getattr(row, field, None)
                    if pd.notnull(value):
                        node_id = g.add_node(str(value).upper())
                        g.add_relation(Relation(service_node_id, node_id, rel_type))
                        if cat_key:
                            g.add_relation(Relation(node_id, category_nodes[cat_key], 'is_a'))
        return g
        
# Usage
graph = Graph.build_from_geo_csv('dmsa_gsp_geo_202510030955.csv')
svc_graph = Graph.build_from_svc_csv('dmsa_its_svc_202510031505.csv')
graph.merge(svc_graph)

@dataclass
class Entity():
    value: str            # Original text from user query
    category: str         # Most specific entity type
    confidence: float     # Confidence score (0.0-1.0)
    context: str          # Reasoning for assignment
    standard_value: str   # Canonical value from knowledge graph/search
    root_entity: str      # Top-level entity type
    role: str             # Role assigned by LLM

class KeyWordsExtraction(Agent):
    def __init__(
        self,
        prompt_templates: Dict[str, str],
        chat_client: ChatClient,
        tools: Optional[List['Tool']] = None,
        agent_ops: Optional[LangfuseOps] = None,
        report_message : Callable[[LogMessage], Awaitable[None]] = None,
    ):
        self._entities: List[Entity] = []
        
        def add_validated_entity(
            value: Annotated[str, "Value as in user query"],
            category: Annotated[str, "Entity category (e.g., pol, pod, port_code, location, service)."],
            confidence: Annotated[float, "Confidence score (0.0-1.0) for the entity assignment."],
            context: Annotated[str, "Short reasoning or context for this assignment."],
            standard_value: Annotated[str, "Canonical/normalized value (UPPERCASE for codes/names)."],
            root_entity: Annotated[Optional[str], "Top-level entity type (e.g., PORT, SERVICE)."] = None,
            role: Annotated[Optional[str], "Role"] = None
        ) -> dict:
            """
            Add a validated logistics entity, including its value, standard_value, category,
            confidence score, context reasoning, and assigned role.
            """
            self._entities.append(Entity(value, category, float(confidence), context, standard_value, root_entity, role))
            return {"status": "added"}

        def complete(
            thought: Annotated[str, "Summary reasoning or explanation for extraction results."],
            rewritten_query: Annotated[str, "Rewritten user query with all validated entities, role, and relevant search results."]
        ) -> dict:
            """
            Return the final extraction result, summary reasoning, and a rewritten user query
            with all validated information.
            """
            return {"status": "ok"}
        
        async def search(
            keyword: Annotated[str, "Keyword to search for."],
            fuzzy: Annotated[bool, "Enable fuzzy matching."] = False,
            threshold: Annotated[int, "Fuzzy match threshold (0-100)."] = 80,
            limit: Annotated[int, "Maximum number of results."] = 10,
            max_hops: Annotated[int, "Maximum graph traversal hops."] = 3
        ) -> Annotated[List[Dict[str, Any]], "List of matching logistics entities."]:
            """
            Search the knowledge graph for logistics entities that match a given keyword.
            Supports fuzzy matching for misspellings and traversal controls to find related entities.
            """
            global graph
            return await graph.search(keyword, fuzzy=fuzzy, threshold=threshold, limit=limit, max_hops=max_hops)
    
        # --- Tool Meta ---
        ESSENTIAL_TOOLS = [
            ("complete", complete, "Return the final extraction result, summary reasoning, and a rewritten user query with all validated information."),
            ("add_validated_entity", add_validated_entity, "Add a validated logistics entity, including its value, standard_value, category, confidence score, context reasoning, and assigned role."),
            ("search", search, "Search the knowledge graph for logistics entities that match a given keyword. Supports fuzzy matching for misspellings and traversal controls to find related entities."),
        ]

        tools = tools or []
        tool_names = {tool.name for tool in tools}
        for name, func, desc in ESSENTIAL_TOOLS:
            if name not in tool_names:
                tools.append(FunctionTool(func, desc, name))

        super().__init__(
            name=KEYWORDS_EXTRACTION_TOPIC,
            description="Extracting, validating, and assigning role to key logistics entities from user queries",
            chat_client=chat_client,
            prompt_templates=prompt_templates,
            agent_ops=agent_ops,
            report_message=report_message,
            tools=tools
        )
        self._max_call_count = 10
        
    @staticmethod
    async def register_agent(
        runtime: SingleThreadedAgentRuntime,
        prompt_templates: Dict[str, str],
        chat_client: ChatClient,
        tools: Optional[List[FunctionTool]] = None,
        agent_ops: Optional[LangfuseOps] = None,
        report_message: Optional[Callable[[LogMessage], Awaitable[None]]] = None,
    ) -> "KeyWordsExtraction":
        return await KeyWordsExtraction.register(
            runtime,
            type=KEYWORDS_EXTRACTION_TOPIC,
            factory=lambda: KeyWordsExtraction(
                prompt_templates=prompt_templates,
                tools=tools,
                chat_client=chat_client,
                agent_ops=agent_ops,
                report_message=report_message
            ),
        )   
            
    @message_handler
    async def on_request(
        self,
        message: KeywordsExtractionRequest,
        ctx: MessageContext
    ) -> Optional[KeywordsExtractionResponse]:
        self._chat_history = []
        self._entities = []
        
        prompt = self.get_prompt(KEYWORDS_EXTRACTION_REQUEST, {})
        self._chat_history.append(SystemMessage(content=prompt))

        user_msg = next(
            (UserMessage(content=msg["content"], source=self.id.key)
             for msg in message.messages if msg["role"] == "user"),
            None
        )
        if not user_msg:
            logger.error("No user message found.")
            response = KeywordsExtractionResponse(
                keywords=[]
            )
            return response

        llm_out: CreateResult = await self.generate(ctx, new_message=user_msg, append_generated_message=False,session_id=self.get_id())


        call_count = self._max_call_count
        done, result = await self._handle_llm_out(llm_out, ctx)
        while not done and call_count > 0:
            call_count -= 1
            llm_out = await self.generate(
                ctx,
                append_generated_message=False,
                session_id=self.get_id(),
            )
            done, result = await self._handle_llm_out(llm_out, ctx)

        keywords = []
        for en in self._entities:
            keywords.append({"category": en.category, "value": en.standard_value, "role": en.role})

        response = KeywordsExtractionResponse(
            keywords=keywords
        )
        
        return response
        
    async def _handle_llm_out(
        self, llm_result: CreateResult, ctx: MessageContext
    ) -> Tuple[bool, Any]:
        """
        Handles the result from the LLM and invokes tools if needed.
        Returns (done, reply_result, is_success).
        """
        if not isinstance(llm_result.content, list) or not all(isinstance(call, FunctionCall) for call in llm_result.content):
            reply_result = "Invalid format: Only function calls are allowed. Please retry with the correct format."
            self._chat_history.extend([
                AssistantMessage(content=llm_result.content, source=self.id.type),
                UserMessage(content=reply_result, source=self.id.type)
            ])
            return False, None

        tools_dict = {tool.name: tool for tool in self._tools}
        tool_calls = []
        
        for call in llm_result.content:
            tool = tools_dict.get(call.name)
            if not tool:
                reply_result = f"Unknown tool: {call.name}"
                self._chat_history.extend([
                    AssistantMessage(content=llm_result.content, source=self.id.type),
                    UserMessage(content=reply_result, source=self.id.type)
                ])
                return False, None
            try:
                arguments = json.loads(call.arguments)
            except json.JSONDecodeError as e:
                reply_result = f"Invalid JSON for tool '{call.name}': {e}"
                self._chat_history.extend([
                    AssistantMessage(content=llm_result.content, source=self.id.type),
                    UserMessage(content=reply_result, source=self.id.type)
                ])
                return False, None
            tool_calls.append((arguments, tool, call))
            
        # Run all tools concurrently
        tool_call_results: List[FunctionExecutionResult] = await asyncio.gather(
            *[run_tool_with_retries(args, tool, call, ctx, self._agent_ops, self.get_id(), None) for args, tool, call in tool_calls]
        )

        # Update chat history once
        self._chat_history.extend([
            AssistantMessage(content=llm_result.content, source=self.id.type),
            FunctionExecutionResultMessage(content=tool_call_results)
        ])

        # Find if any tool call signals done/failed
        for result in tool_call_results:
            if result.name == "complete":
                content = result.content.replace("'", "\"")
                return True, json.loads(content)

        return False, None