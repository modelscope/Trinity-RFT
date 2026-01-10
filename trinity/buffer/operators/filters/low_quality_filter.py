from typing import List, Tuple
import math

from trinity.buffer.operators.experience_operator import ExperienceOperator
from trinity.common.experience import Experience


class LowQualityExperienceFilter(ExperienceOperator):
    def process(self, exps: List[Experience]) -> Tuple[List[Experience], dict]:
        kept = [e for e in exps if e.reward is not None and e.reward == e.reward]

        return kept, {"filtered_count": len(exps) - len(kept)}
