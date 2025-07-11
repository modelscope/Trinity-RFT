from typing import Any, Dict, List


def to_rm_gallery_messages(messages: List[Dict[str, Any]]) -> Any:
    """
    Converts string list to structured ChatMessage list for debugging.

    Args:
        messages: List of alternating user/assistant messages

    Returns:
        List of structured ChatMessage objects
    """
    from rm_gallery.core.model.message import ChatMessage, MessageRole

    role_map = {
        "system": MessageRole.SYSTEM,
        "user": MessageRole.USER,
        "assistant": MessageRole.ASSISTANT,
    }

    return [ChatMessage(role=role_map[msg["role"]], content=msg["content"]) for msg in messages]
