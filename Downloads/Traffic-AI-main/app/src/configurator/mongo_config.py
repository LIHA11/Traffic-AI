from typing import TypedDict


class MongoConfig(TypedDict):
    service_name: str
    conversation_collection: str
