from typing import List, Tuple
from src.conversations.vo.message import Message
from src.conversations.enum.role_enum import RoleEnum
import json

def get_eval_set(path: str) -> List[Tuple[List[Message], str]]:
    """
    Loads evaluation set from a JSON file.
    Each section contains items with 'messages' and 'expectation'.
    """
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Data should be a dictionary with sections as keys.")

    role_map = RoleEnum._value2member_map_
    eval_set = [
        (get_sample_from_dict(item, role_map), item['expectation'])
        for items in data.values()
        for item in items
        if 'expectation' in item
    ]
    return eval_set

def get_sample_from_dict(item: dict, role_map=None) -> List[Message]:
    """
    Converts a dict item to a list of Message objects.
    """
    if role_map is None:
        role_map = RoleEnum._value2member_map_
    messages = item.get('messages', [])
    if not isinstance(messages, list):
        raise ValueError("'messages' should be a list.")

    return [
        Message.from_dict(message)
        if (
            isinstance(message, dict) and
            'role' in message and
            'content' in message and
            message['role'] in role_map
        ) else (_ for _ in ()).throw(ValueError("Invalid message or role not found"))
        for message in messages
    ]
