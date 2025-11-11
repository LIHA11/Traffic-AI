from autogen_core.tools import FunctionTool
from typing_extensions import Annotated


class Handoff(FunctionTool):

    def __init__(self, topic: str, description: str, **kwargs) -> None:
        def handoff_func(thought: Annotated[str, "Brief reasoning or thought process"]):
            return topic

        super().__init__(
            func=handoff_func,
            description=description,
            name=topic,
            **kwargs
        )
