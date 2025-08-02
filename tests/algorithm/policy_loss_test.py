# -*- coding: utf-8 -*-
"""Test for policy loss functions"""

import unittest

import torch
from verl import DataProto

from trinity.algorithm.policy_loss_fn.policy_loss_fn import POLICY_LOSS_FN


class VerlPolicyLossTest(unittest.TestCase):
    def setUp(self):
        seed = 42
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        shape = (5, 20)
        self.logprob = 2 * torch.rand(shape) - 1
        self.input_data = DataProto.from_dict(
            {
                "old_log_probs": 2 * torch.rand(shape) - 1,
                "ref_log_prob": 2 * torch.rand(shape) - 1,
                "response_mask": torch.rand(shape) > 0.5,
                "advantages": 2 * torch.rand(shape) - 1,
                "expert_mask": torch.rand(shape[0]) > 0.5,
            }
        )

    def test_ppo_policy_loss(self):
        policy_loss_fn_cls = POLICY_LOSS_FN.get("ppo")
        policy_loss_fn_args = policy_loss_fn_cls.default_args()
        policy_loss_fn = policy_loss_fn_cls(**policy_loss_fn_args)
        loss, metrics = policy_loss_fn(log_prob=self.logprob, **self.input_data.batch)
        ppo_loss = torch.tensor(0.28560468554496765)
        pg_clipfrac = torch.tensor(0.3541666567325592)
        ppo_kl = torch.tensor(-0.21663446724414825)
        self.assertTrue(torch.allclose(loss, ppo_loss))
        self.assertTrue(torch.allclose(torch.tensor(metrics["pg_clipfrac"]), pg_clipfrac))
        self.assertTrue(torch.allclose(torch.tensor(metrics["ppo_kl"]), ppo_kl))
        self.assertTrue(torch.allclose(torch.tensor(metrics["pg_loss"]), ppo_loss))

    def test_gspo_policy_loss(self):
        policy_loss_fn_cls = POLICY_LOSS_FN.get("gspo")
        policy_loss_fn_args = policy_loss_fn_cls.default_args()
        policy_loss_fn = policy_loss_fn_cls(**policy_loss_fn_args)
        loss, metrics = policy_loss_fn(log_prob=self.logprob, **self.input_data.batch)
        gspo_loss_expected = torch.tensor(0.27235108613967896)
        pg_clipfrac_expected = torch.tensor(0.375)
        ppo_kl_seq_expected = torch.tensor(-0.21027061343193054)
        ppo_kl_expected = torch.tensor(-0.21663446724414825)
        print(f"{loss.item()=}, {metrics=}")
        self.assertTrue(torch.allclose(loss, gspo_loss_expected))
        self.assertTrue(torch.allclose(torch.tensor(metrics["pg_clipfrac"]), pg_clipfrac_expected))
        self.assertTrue(torch.allclose(torch.tensor(metrics["ppo_kl_seq"]), ppo_kl_seq_expected))
        self.assertTrue(torch.allclose(torch.tensor(metrics["ppo_kl"]), ppo_kl_expected))
        self.assertTrue(torch.allclose(torch.tensor(metrics["pg_loss"]), gspo_loss_expected))

    def test_sft_policy_loss(self):
        policy_loss_fn_cls = POLICY_LOSS_FN.get("sft")
        policy_loss_fn_args = policy_loss_fn_cls.default_args()
        policy_loss_fn = policy_loss_fn_cls(**policy_loss_fn_args)
        loss, metrics = policy_loss_fn(log_prob=self.logprob, **self.input_data.batch)
        sft_loss = torch.tensor(-0.07560186833143234)
        self.assertTrue(torch.allclose(loss, sft_loss))
        self.assertTrue(torch.allclose(torch.tensor(metrics["sft_loss"]), sft_loss))

    def test_dpo_policy_loss(self):
        policy_loss_fn_cls = POLICY_LOSS_FN.get("dpo")
        policy_loss_fn_args = policy_loss_fn_cls.default_args()
        policy_loss_fn = policy_loss_fn_cls(**policy_loss_fn_args)
        loss, metrics = policy_loss_fn(log_prob=self.logprob, **self.input_data.batch)
        dpo_loss = torch.tensor(0.5406752228736877)
        chosen_reward = torch.tensor(0.7082431316375732)
        rejected_reward = torch.tensor(0.3757950782775879)
        accuracy_mean = torch.tensor(1.0)
        self.assertTrue(torch.allclose(loss, dpo_loss))
        self.assertTrue(torch.allclose(torch.tensor(metrics["chosen_reward"]), chosen_reward))
        self.assertTrue(torch.allclose(torch.tensor(metrics["rejected_reward"]), rejected_reward))
        self.assertTrue(torch.allclose(torch.tensor(metrics["accuracy_mean"]), accuracy_mean))
        self.assertTrue(torch.allclose(torch.tensor(metrics["dpo_loss"]), dpo_loss))

    def test_opmd_policy_loss(self):
        policy_loss_fn_cls = POLICY_LOSS_FN.get("opmd")
        policy_loss_fn_args = policy_loss_fn_cls.default_args()
        policy_loss_fn = policy_loss_fn_cls(**policy_loss_fn_args)
        loss, metrics = policy_loss_fn(log_prob=self.logprob, **self.input_data.batch)
        opmd_loss = torch.tensor(-0.009589947760105133)
        self.assertTrue(torch.allclose(loss, opmd_loss))
        self.assertTrue(torch.allclose(torch.tensor(metrics["opmd_loss"]), opmd_loss))

    def test_mix_policy_loss(self):
        policy_loss_fn_cls = POLICY_LOSS_FN.get("mix")
        policy_loss_fn_args = policy_loss_fn_cls.default_args()
        policy_loss_fn = policy_loss_fn_cls(**policy_loss_fn_args)
        loss, metrics = policy_loss_fn(log_prob=self.logprob, **self.input_data.batch)
        mix_loss = torch.tensor(0.6581965088844299)
        pg_clipfrac = torch.tensor(0.7777777910232544)
        ppo_kl = torch.tensor(-1.0737695693969727)
        pg_loss = torch.tensor(0.7236452102661133)
        sft_loss = torch.tensor(0.06915830634534359)
        self.assertTrue(torch.allclose(loss, mix_loss))
        self.assertTrue(torch.allclose(torch.tensor(metrics["usual/pg_clipfrac"]), pg_clipfrac))
        self.assertTrue(torch.allclose(torch.tensor(metrics["usual/ppo_kl"]), ppo_kl))
        self.assertTrue(torch.allclose(torch.tensor(metrics["usual/pg_loss"]), pg_loss))
        self.assertTrue(torch.allclose(torch.tensor(metrics["expert/sft_loss"]), sft_loss))
        self.assertTrue(torch.allclose(torch.tensor(metrics["loss"]), mix_loss))
