# Standard Library Imports
from typing import (
    Dict,
    List,
    Optional, 
    Union
)

# Third-party library imports
from pydantic import BaseModel

PREFERENCE_EXTRACTOR_MASTER_TOPIC = "preference_extractor_master"

class PreferenceExtractionMasterRequest(BaseModel):
    message_id: Optional[str] = None
    user_id: Optional[str] = None
    first_n_conversations: Optional[int] = None

class PreferenceExtractionMasterResponse(BaseModel):
    summary: str

MESSAGE_GROUPER_REQUEST = "message_grouper_request"
MESSAGE_GROUPER_TOPIC = "message_grouper"

class MessageGrouperRequest(BaseModel):
    messages: List[dict]
    messages_groups: Dict[str, Union[str, None]]
    refresh_all_msgs: Optional[bool] = False

class MessageGrouperResponse(BaseModel):
    messages_by_groups: Optional[List[List[dict]]] = None

KEYWORDS_EXTRACTION_TOPIC = "keywords_extraction"
KEYWORDS_EXTRACTION_REQUEST = "keywords_extraction_request"

class KeywordsExtractionRequest(BaseModel):
    messages: List[dict]

class KeywordsExtractionResponse(BaseModel):
    keywords: Optional[List[Dict[str, str]]] = None
    
SUMMARY_EXTRACTION_TOPIC = "summary_extraction"
SUMMARY_EXTRACTION_REQUEST = "summary_extraction_request"

class SummaryExtractionRequest(BaseModel):
    messages: List[dict]

class SummaryExtractionResponse(BaseModel):
    summary: Optional[str] = None

