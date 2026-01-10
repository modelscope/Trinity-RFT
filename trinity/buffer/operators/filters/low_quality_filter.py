from typing import List, Tuple
import math

from trinity.buffer.operators.experience_operator import ExperienceOperator
from trinity.common.experience import Experience


class LowQualityExperienceFilter(ExperienceOperator):
    def process(self, exps: List[Experience]) -> Tuple[List[Experience], dict]:
        kept = []
        for e in exps:
            r = e.reward
            if r is None:
                continue
            if isinstance(r, float) and math.isnan(r):
                continue
            kept.append(e)

        return kept, {"filtered_count": len(exps) - len(kept)}
