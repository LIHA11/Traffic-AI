from typing import TypedDict, Dict


class MongoCreds(TypedDict):
    port: int
    user: str
    password: str
    database: str
    hosts: Dict[str, str]
    replica_set: str
