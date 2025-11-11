from typing import Dict, Optional
from mongoengine import connect
from pymongo import MongoClient

class MongoDB:
    def __init__(
        self,
        user: str,
        password: str,
        hosts: Dict[str, str],
        port: int,
        database: str,
        conversations_collection: str,
        replica_set: Optional[str] = None,
    ):
        host_list = ",".join([f"{host}:{port}" for host in hosts.values()])
        auth_str = f"mongodb://{user}:{password}@{host_list}/{database}"
        if replica_set:
            auth_str += f"?replicaSet={replica_set}"

        self.database = database
        self.conversations_collection = conversations_collection

        connect(host=auth_str)
class PyMongoDB:
    def __init__(
        self,
        user: str,
        password: str,
        hosts: Dict[str, str],
        port: int,
        database: str,
        conversations_collection: str,
        replica_set: Optional[str] = None,
    ):
        host_list = ",".join([f"{host}:{port}" for host in hosts.values()])
        auth_str = f"mongodb://{user}:{password}@{host_list}/{database}"
        self.client = MongoClient(auth_str)
