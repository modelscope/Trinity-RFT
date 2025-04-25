# -*- coding: utf-8 -*-
"""Experience Class."""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from itertools import chain, repeat
from typing import List, Optional

import torch
from torch import Tensor


@dataclass
class Experience:
    """A single experience."""

    tokens: Tensor  # [seq]
    prompt_length: int
    logprobs: Optional[Tensor] = None  # [seq]
    reward: Optional[float] = None
    prompt_text: Optional[str] = None
    response_text: Optional[str] = None
    action_mask: Optional[Tensor] = None
    chosen: Optional[Tensor] = None  # for dpo
    rejected: Optional[Tensor] = None  # for dpo
    info: Optional[dict] = None
    metrics: Optional[dict[str, float]] = None
    run_id: str = ""

    def __post_init__(self):
        if self.action_mask is not None:
            assert (
                self.action_mask.shape == self.tokens.shape
            ), "The provided action_mask must have the same shape as tokens."

    def serialize(self) -> bytes:
        """Serialize the experience to bytes."""
        return pickle.dumps(self)

    @staticmethod
    def deserialize(data: bytes) -> Experience:
        """Deserialize the experience from bytes."""
        return pickle.loads(data)


@dataclass(frozen=True)
class Experiences:
    """A container for a batch of experiences, for high performance communication usage.

    Example:

        >>>             |<- prompt_length ->|               |
        >>> tokens: ('P' represents prompt, 'O' represents output)
        >>> exp1:       |........PPPPPPPPPPP|OOOOOOOOOO.....|
        >>> exp2:       |......PPPPPPPPPPPPP|OOOOOOO........|
        >>>
        >>> attention_masks: ('.' represents False and '1' represents True)
        >>> exp1:       |........11111111111|1111111111.....|
        >>> exp2:       |......1111111111111|1111111........|
    """

    tokens: Tensor
    rewards: Tensor
    attention_masks: Tensor
    action_masks: Optional[Tensor]
    prompt_length: int
    logprobs: Optional[Tensor]
    run_ids: List[str]

    @property
    def batch_size(self) -> int:
        """Get the batch size."""
        return self.tokens.size(0)

    @classmethod
    def gather_experiences(
        cls, experiences: list[Experience], pad_token_id: int = 0
    ) -> Experiences:
        """Gather a batch of experiences from a list of experiences.

        This method will automatically pad the `tokens` and `logprobs` of input experiences to the same length.
        """
        if len(experiences) == 0:
            return Experiences(
                tokens=torch.empty(0, dtype=torch.int32),
                rewards=torch.empty(0, dtype=torch.float32),
                attention_masks=torch.empty(0, dtype=torch.bool),
                action_masks=torch.empty(0, dtype=torch.bool),
                logprobs=torch.empty(0, dtype=torch.float32),
                prompt_length=torch.empty(0, dtype=torch.int32),
                run_ids=[],
            )
        max_prompt_length = max([exp.prompt_length for exp in experiences])
        max_response_length = max([len(exp.tokens) - exp.prompt_length for exp in experiences])
        run_ids = [exp.run_id for exp in experiences]
        tokens_dtype = experiences[0].tokens.dtype
        tokens = torch.stack(
            [
                torch.cat(
                    [
                        torch.full(
                            (max_prompt_length - exp.prompt_length,),
                            pad_token_id,
                            dtype=tokens_dtype,
                        ),
                        exp.tokens,
                        torch.full(
                            (max_response_length + exp.prompt_length - len(exp.tokens),),
                            pad_token_id,
                            dtype=tokens_dtype,
                        ),
                    ]
                )
                for exp in experiences
            ]
        )
        if experiences[0].reward is not None:
            rewards = torch.tensor([exp.reward for exp in experiences], dtype=torch.float)
        else:
            rewards = None

        # Calculate the action_masks according to the provided experience.action_mask
        if experiences[0].action_mask is not None:
            action_mask_dtype = experiences[0].action_mask.dtype
            action_masks = torch.stack(
                [
                    torch.cat(
                        [
                            torch.full(
                                (max_prompt_length - exp.prompt_length,),
                                0,
                                dtype=action_mask_dtype,
                            ),
                            exp.action_mask,
                            torch.full(
                                (max_response_length + exp.prompt_length - len(exp.tokens),),
                                0,
                                dtype=action_mask_dtype,
                            ),
                        ]
                    )
                    for exp in experiences
                ]
            )
        else:
            action_masks = None
        attention_masks = torch.zeros(
            (len(experiences), max_prompt_length + max_response_length), dtype=torch.bool
        )
        for i, exp in enumerate(experiences):
            start = max_prompt_length - exp.prompt_length
            end = start + len(exp.tokens)
            attention_masks[i, start:end] = 1

        if all(exp.logprobs is not None for exp in experiences):
            logprob_dtype = experiences[0].logprobs.dtype  # type: ignore [union-attr]
            logprobs = torch.stack(
                [
                    torch.cat(
                        [
                            torch.full(
                                (max_prompt_length - exp.prompt_length,),
                                0.0,
                                dtype=logprob_dtype,
                            ),
                            exp.logprobs,
                            torch.full(
                                (max_response_length + exp.prompt_length - len(exp.tokens),),
                                0.0,
                                dtype=logprob_dtype,
                            ),
                        ]
                    )
                    for exp in experiences
                ]
            )
        else:
            logprobs = None

        return cls(
            run_ids=run_ids,
            tokens=tokens,
            rewards=rewards,
            attention_masks=attention_masks,
            action_masks=action_masks,
            prompt_length=max_prompt_length,
            logprobs=logprobs,
        )

    @classmethod
    def gather_dpo_experiences(
        cls, experiences: list[Experience], pad_token_id: int = 0
    ) -> Experiences:
        """Gather a batch of dpo experiences from a list of experiences.

        Reference: https://github.com/huggingface/trl/blob/main/trl/trainer/dpo_trainer.py#L849

        Note: We arrange inputs in the order of (chosen, rejected, chosen, rejected, ...)
                to ensure that each pair of (chosen, rejected) is not split by subsequent operations

        Args:
            Experiences: `(list[Experience])`
                - `"prompt"`: token ids of the prompt
                - `"chosen"`: token ids of the chosen response
                - `"rejected"`: token ids of the rejected response
            pad_token_id: `(int)`
                The pad token id.

        Returns:
            Experiences:
                - `"tokens"`: Concatenated chosen and rejected completion input IDs of shape `(2 * batch_size, max_completion_length)`.
                - `"attention_masks"`: Concatenated chosen and rejected attention masks of shape `(2 * batch_size, max_completion_length)`.
        """
        if len(experiences) == 0:
            return Experiences(
                tokens=torch.empty(0, dtype=torch.int32),
                rewards=torch.empty(0, dtype=torch.float32),
                attention_masks=torch.empty(0, dtype=torch.bool),
                action_masks=torch.empty(0, dtype=torch.bool),
                logprobs=torch.empty(0, dtype=torch.float32),
                prompt_length=torch.empty(0, dtype=torch.int32),
                run_ids=[],
            )

        # TODO: exp.tokens in DPO are prompt tokens
        prompt_tokens = list(chain.from_iterable([repeat(exp.tokens, 2) for exp in experiences]))
        max_prompt_length = max([exp.prompt_length for exp in experiences])

        chosen_tokens = [exp.chosen for exp in experiences]
        rejected_tokens = [exp.rejected for exp in experiences]
        response_tokens = list(chain.from_iterable(zip(chosen_tokens, rejected_tokens)))
        max_response_length = max([len(response) for response in response_tokens])  # type: ignore

        run_ids = list(chain.from_iterable([repeat(exp.run_id, 2) for exp in experiences]))
        tokens_dtype = experiences[0].tokens.dtype
        tokens = torch.stack(
            [
                torch.cat(
                    [
                        torch.full(
                            (max_prompt_length - len(prompt),),
                            pad_token_id,
                            dtype=tokens_dtype,
                        ),
                        prompt,
                        response,
                        torch.full(
                            (max_response_length - len(response),),  # type: ignore
                            pad_token_id,
                            dtype=tokens_dtype,
                        ),
                    ]
                )
                for prompt, response in zip(prompt_tokens, response_tokens)
            ]
        )

        attention_masks = torch.zeros(
            (len(tokens), max_prompt_length + max_response_length), dtype=torch.bool
        )

        for (i, prompt), response in zip(enumerate(prompt_tokens), response_tokens):
            start = max_prompt_length - len(prompt)
            end = max_prompt_length + len(response)  # type: ignore
            attention_masks[i, start:end] = 1

        assert len(tokens) == 2 * len(experiences)

        return cls(
            run_ids=run_ids,
            tokens=tokens,
            attention_masks=attention_masks,
            prompt_length=max_prompt_length,
            rewards=None,
            action_masks=None,
            logprobs=None,
        )
