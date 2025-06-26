# -*- coding: utf-8 -*-
"""Test cases for Storage modules."""
import os
import unittest

import torch

from trinity.common.experience import Experience, Experiences
from trinity.buffer.schema.sql_schema import ExperienceModel

db_url = os.path.join(os.path.dirname(__file__), "tmp", "test.db")
dataset_path = os.path.join(os.path.dirname(__file__), "data")


class TestExperienceConversion(unittest.TestCase):
    """Test cases for ExperienceModel"""

    def test_experience_model_experience_conversion(self):
        """Test the conversion between Experience and ExperienceModel"""
        tokens = torch.tensor([1, 2, 3], dtype=torch.int32)
        reward = 0.6
        prompt_length = 2
        logprobs = torch.tensor([0, 0, 0.1], dtype=torch.float32)
        action_mask = torch.tensor([1, 0, 1], dtype=torch.bool)
        experience = Experience(
            tokens=tokens,
            reward=reward,
            prompt_length=prompt_length,
            logprobs=logprobs,
            action_mask=action_mask,
        )

        model = ExperienceModel.from_experience(experience)
        experience = model.to_experience()
        self.assertTrue(torch.equal(experience.tokens, tokens))
        self.assertEqual(experience.prompt_length, prompt_length)
        self.assertEqual(experience.reward, reward)
        self.assertTrue(torch.equal(experience.logprobs, logprobs))
        self.assertTrue(torch.equal(experience.action_mask, action_mask))

    def test_batch_conversion(self):
        exps = [
            Experience(
                tokens=torch.tensor([1, 2]),
                prompt_length=1,
                reward=float(0.1),
                logprobs=torch.tensor([0, 0.1]),
                action_mask=torch.tensor([1, 0]),
            ),
            Experience(
                tokens=torch.tensor([1, 2, 3]),
                prompt_length=2,
                reward=float(0.2),
                logprobs=torch.tensor([0, 0, 0.1]),
                action_mask=torch.tensor([1, 0, 1]),
            ),
            Experience(
                tokens=torch.tensor([1, 2, 3, 4]),
                prompt_length=2,
                reward=float(0.3),
                logprobs=torch.tensor([0, 0, 0.1, 0.2]),
                action_mask=torch.tensor([1, 0, 1, 0]),
            ),
            Experience(
                tokens=torch.tensor([1, 2, 3, 4]),
                prompt_length=3,
                reward=float(0.4),
                logprobs=torch.tensor([0, 0, 0, 0.1]),
                action_mask=torch.tensor([1, 0, 1, 0]),
            ),
        ]
        batch = Experiences.gather_experiences(exps)
        self.assertEqual(batch.batch_size, 4)
        self.assertEqual(batch.prompt_length, 3)
        prompt_length = batch.prompt_length
        for i in range(batch.batch_size):
            self.assertEqual(batch.rewards[i], exps[i].reward)
            self.assertTrue(
                torch.all(
                    batch.tokens[i][
                        prompt_length
                        - exps[i].prompt_length : prompt_length
                        - exps[i].prompt_length
                        + exps[i].tokens.size(0)
                    ]
                    == exps[i].tokens
                )
            )
            self.assertTrue(
                torch.all(
                    batch.logprobs[i][
                        prompt_length
                        - exps[i].prompt_length : prompt_length
                        + exps[i].tokens.size(0)
                        - exps[i].prompt_length
                    ]
                    == exps[i].logprobs
                )
            )
            self.assertTrue(
                torch.all(
                    batch.action_masks[i][
                        prompt_length
                        - exps[i].prompt_length : prompt_length
                        - exps[i].prompt_length
                        + exps[i].action_mask.size(0)
                    ]
                    == exps[i].action_mask
                )
            )


if __name__ == "__main__":
    unittest.main()
