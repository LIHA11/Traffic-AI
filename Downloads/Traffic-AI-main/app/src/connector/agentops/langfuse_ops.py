import uuid
from typing import Optional, Tuple

from langfuse import Langfuse
from src.connector.agentops.agentops import AgentOps

class LangfuseOps(AgentOps):
    def __init__(self, secret_key: str, public_key: str, host: str) -> None:
        self.instance = Langfuse(
            secret_key=secret_key,
            public_key=public_key,
            host=host,
        )

    def run(self, *args, **kwargs) -> Tuple[Optional[str]]:
        trace_id = kwargs.get('id') or str(uuid.uuid4())
        kwargs['id'] = trace_id
        return self.instance.trace(*args, **kwargs), trace_id
    
    def end_trace(self, id: str) -> None:
        pass
    
