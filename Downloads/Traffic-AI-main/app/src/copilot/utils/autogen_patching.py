# autogen_patching.py
import importlib
import sys
from types import ModuleType
from typing import Type

# Define the target module and class name constants
TARGET_MODULE = "mlflow.types.chat"
ORIGINAL_CLASS_NAME = "ChatMessage"

try:
    original_module = importlib.import_module(TARGET_MODULE)
    BaseModel = getattr(original_module, "BaseModel")
    OriginalChatMessage = getattr(original_module, ORIGINAL_CLASS_NAME)

    from typing import Union, List, Optional, Dict, Any

    content_type = Union[str, List[Any], Dict[str, Any], None]

    class PatchedChatMessage(BaseModel):
        """
        Monkey-patched version of mlflow.types.chat.ChatMessage
        to allow dict type for content.
        """
        role: str
        content: content_type = None

except (ImportError, AttributeError) as e:
    print(f"Warning: Could not prepare patch for {TARGET_MODULE}.{ORIGINAL_CLASS_NAME}. Error: {e}")
    PatchedChatMessage = None
    OriginalChatMessage = None

class ClassReplacer:
    def __init__(
            self,
            target_module: str = TARGET_MODULE,
            original_class_name: str = ORIGINAL_CLASS_NAME,
            new_class: Type = None,
    ):
        self._target_module = target_module
        self._original_class_name = original_class_name
        self._new_class = new_class

        self._module = importlib.import_module(self._target_module)
        self._original_class = getattr(self._module, original_class_name, None)

    def apply(self):
        if not self._original_class or not self._new_class:
             print("Warning: ClassReplacer not properly initialized. Skipping patch.")
             return

        patched_count = 0
        for mod_name, mod in list(sys.modules.items()):
            if mod is None or not isinstance(mod, ModuleType):
                continue

            if hasattr(mod, self._original_class_name):
                current_ref = getattr(mod, self._original_class_name)
                if current_ref is self._original_class:
                    setattr(mod, self._original_class_name, self._new_class)
                    patched_count += 1

        print(f"Applied monkey patch: Replaced {patched_count} references of {self._original_class_name}.")


def apply_mlflow_autogen_patch():
    """Applies the monkey patch for ChatMessage."""
    if PatchedChatMessage is None:
         print("Error: Cannot apply patch, PatchedChatMessage was not created successfully.")
         return

    replacer = ClassReplacer(
        target_module=TARGET_MODULE,
        original_class_name=ORIGINAL_CLASS_NAME,
        new_class=PatchedChatMessage,
    )
    replacer.apply()
        