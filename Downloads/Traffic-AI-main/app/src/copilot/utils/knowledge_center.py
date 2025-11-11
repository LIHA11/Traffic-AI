import httpx
import logging
import asyncio
import json

from typing import List, Optional
from pydantic import BaseModel
from typing_extensions import Annotated
from autogen_core.tools import FunctionTool

from src.dto.knowledge_center_user_preferences_response_dto import KnowledgeCenterUserPreferenceResponseDto

logger = logging.getLogger(__name__)

class Document(BaseModel):
    content: str
    title: str
    similarity: float
    id: int
    
class KnowledgeCenter:
    
    MAX_RETRIES = 10  # Maximum number of retries
    RETRY_DELAY = 2.0  # Delay between retries in seconds
    
    def __init__(self, knowledge_center_url: str):
        self._knowledge_center_url = knowledge_center_url
        self._tools = [
            FunctionTool(
                self.search_domain_knowledge,
                name="search_domain_knowledge",
                description="""
Searches the knowledge base for documents most relevant to the provided query. Returns the top N documents ranked by similarity. The documents are ranked by similarity, each including a title, content snippet, and similarity score.",
(1 = highest similarity, 0 = lowest similarity).",
The document title reflects the main topic, while the content provides more detailed information about that subject",
Note: Returned documents are simply the most similar available and may not always be directly relevant or having to the query."         
"""
            ),   
        ]
        
    def get_tools(self) -> List[FunctionTool]:
        return self._tools
        
    async def get_documents(
        self,
        query_text: str,
        search_type: str = "title",
        num_results: int = 2,
        domains: Optional[str] = None,
        title_weight: float = 0.5,
        distance_method: str = "cosine"
    ) -> List["Document"]:
        """
        Retrieve demonstration documents from the knowledge center.
        """
        params = {
            "query_text": query_text,
            "search_type": search_type,
            "num_results": num_results,
            "title_weight": title_weight,
            "distance_method": distance_method,
        }
        if domains:
            params["domains"] = domains

        async with httpx.AsyncClient(verify=False, timeout=100.0) as client:
            for attempt in range(1, KnowledgeCenter.MAX_RETRIES + 1):
                try:
                    response = await client.get(
                        f"{self._knowledge_center_url}/documents/embeddings/search",
                        params=params
                    )
                    if response.status_code == 404:
                        # Fallback: local mock may not implement the remote path; try /query
                        logger.warning("/documents/embeddings/search 404. Falling back to /query endpoint.")
                        try:
                            q_payload = {"text": query_text, "top_k": num_results}
                            q_resp = await client.post(f"{self._knowledge_center_url}/query", json=q_payload)
                            q_resp.raise_for_status()
                            raw_results = q_resp.json().get("results", [])
                            # Wrap in Document structure heuristic
                            return [
                                Document(id=i, content=txt, title=(txt.strip().split("\n")[0][:64] or "Untitled"), similarity=1.0)
                                for i, txt in enumerate(raw_results)
                            ]
                        except Exception as fe:
                            logger.error(f"Fallback /query failed: {fe}")
                            q_resp.raise_for_status()  # re-raise original error path
                    response.raise_for_status()
                    documents = response.json().get("documents", [])
                    return [
                        Document(id=doc["document"].get("DocumentID", ""), content=doc["document"].get("Content", ""), title=doc["document"].get("DocumentTitle", ""), similarity=doc.get("similarity", 0.0))
                        for doc in documents
                    ]
                except httpx.HTTPStatusError as e:
                    logger.error("HTTP error on attempt %d: %s", attempt, e, exc_info=False)
                except Exception as e:
                    logger.error(f"Error retrieving documents on attempt {attempt}: {e}")
                
                if attempt < KnowledgeCenter.MAX_RETRIES:
                    logger.error("Retrying in %d seconds...", KnowledgeCenter.RETRY_DELAY)
                    await asyncio.sleep(KnowledgeCenter.RETRY_DELAY)
                else:
                    logger.error("Max retries reached. Returning an empty list.")
                    return []
                
    async def search_domain_knowledge_basic(
        self, 
        query_text: Annotated[str, "Keywords for retrieving domain knowledge from the knowledge base"],
    ) -> str:
        NUM_OF_DOCS, TITLE_WEIGHT, DOMAIN, SERACH_TYPE = 5, 0.7, "domain_knowledge", "combined"
        
        DOMAIN = DOMAIN + "," + "basic"
        
        docs = await self.get_documents(
            query_text, 
            num_results=NUM_OF_DOCS, 
            domains=DOMAIN, 
            search_type=SERACH_TYPE, 
            title_weight=TITLE_WEIGHT
        )
        if len(docs) == 0:
            return json.dumps({
                "results": [],
                "message": "No relevant domain knowledge found."
            })
        
        results = []
        for i, doc in enumerate(docs):
            results.append({
                "id": doc.id,
                "document title": doc.title,
                "document content": doc.content.strip(), 
                "similarity": doc.similarity
            })
        
        return json.dumps({
            "results": results,
            "count": len(results)
        }, ensure_ascii=False, indent=2)
        
    async def search_domain_knowledge(
        self, 
        thought: Annotated[str, "Brief explanation"],
        query_text: Annotated[str, "Keywords for retrieving domain knowledge from the knowledge base"],
        #short_code_mode: Annotated[Optional[bool], "If True, restrict the search to short code definitions only (e.g., BBB, CCC-DD-EE-FF), default = False"] = False
    ) -> str:
        
        NUM_OF_DOCS, TITLE_WEIGHT, DOMAIN, SERACH_TYPE = 5, 0.5, "domain_knowledge", "combined"
        
        logging.error("search_domain_knowledge called with thought: %s, query_text: %s, short_code_mode: %s, domain: %s", thought, query_text, DOMAIN)
        
        docs = await self.get_documents(
            query_text, 
            num_results=NUM_OF_DOCS, 
            domains=DOMAIN, 
            search_type=SERACH_TYPE, 
            title_weight=TITLE_WEIGHT
        )
        if len(docs) == 0:
            return json.dumps({
                "results": [],
                "message": "No relevant domain knowledge found."
            })
        
        results = []
        for i, doc in enumerate(docs):
            results.append({
                "id": doc.id,
                "document content": doc.content.strip(), 
                "similarity": doc.similarity
            })
        
        return json.dumps({
            "results": results,
            "count": len(results)
        }, ensure_ascii=False, indent=2)

    async def insert_keywords(self, keywords: List[str], message_group_id: str, user_id: str, old_message_group_ids: List[str]) -> bool:
        url = f"{self._knowledge_center_url}/preferences/tags/insert"
        payload = {
            "tags": keywords,
            "message_group_id": message_group_id,
            "user_id": user_id,
            "old_message_group_ids": old_message_group_ids
        }
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(url, json=payload)
                if response.status_code != 200:
                    return False
                data = response.json()
                if not data.get("success", False):
                        return False
        except Exception as e:
            logger.error(f"Error inserting keywords: {e}")
            return False
    
    async def insert_summary(self, summary: str, message_group_id: str, user_id: str, old_message_group_ids: List[str]) -> bool:
        url = f"{self._knowledge_center_url}/preferences/summary/insert"
        payload = {
            "summary": summary,
            "message_group_id": message_group_id,
            "user_id": user_id,
            "old_message_group_ids": old_message_group_ids
        }
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(url, json=payload)
                if response.status_code != 200:
                    return False
                data = response.json()
                if not data.get("success", False):
                    return False
        except Exception as e:
            logger.error(f"Error inserting summary: {e}")
            return False

    async def get_user_preferences(self, user_id: str, query_text: str, threshold: float = 0, top_n: int = 10) -> KnowledgeCenterUserPreferenceResponseDto:
        url = f"{self._knowledge_center_url}/preferences/search"
        params = {
            "user_id": user_id,
            "query_text": query_text,
            "threshold": threshold,
            "top_n": top_n
        }
        try:
            async with httpx.AsyncClient(timeout=None) as client:
                logger.info(f"Get user preferences params: {json.dumps(params)}")
                response = await client.get(url, params=params)
                if response.status_code != 200:
                    return KnowledgeCenterUserPreferenceResponseDto(error=f"HTTP error: {response.status_code}", results=[])
                data = response.json()
                logger.info(f"Get user preferences data: {json.dumps(data)}")
                return KnowledgeCenterUserPreferenceResponseDto(error=data.get("error", None), results=data.get("results", []))
        except Exception as e:
            logger.error(f"Error getting user preferences: {e}")
            return KnowledgeCenterUserPreferenceResponseDto(error=str(e), results=[])