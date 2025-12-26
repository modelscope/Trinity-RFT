# -*- coding: utf-8 -*-
"""Key Mapper"""

from typing import Dict


class KeyMapper:
    def __init__(self, to_trinity_map: Dict[str, str]):
        self.to_trinity_map = to_trinity_map
        self.from_trinity_map = {v: k for k, v in self.to_trinity_map.items()}

    def to_trinity(self, key: str) -> str:
        return self.to_trinity_map.get(key, key)

    def from_trinity(self, key: str) -> str:
        return self.from_trinity_map.get(key, key)


ALL_MAPPERS = {
    "verl": KeyMapper(
        {
            "log_prob": "logprob",
            "old_log_probs": "old_logprob",
            "ref_log_prob": "ref_logprob",
            "response_mask": "action_mask",
            "advantages": "advantages",
        }
    ),
    "tinker": KeyMapper({}),
}
