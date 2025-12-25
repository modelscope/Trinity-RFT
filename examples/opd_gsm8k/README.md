# Example: On-Policy Distillation on GSM8K dataset

This example demonstrates On-Policy Distillation (OPD) algorithm training on the GSM8K dataset.

On-Policy Distillation is a knowledge distillation method, where in this example:
1. **Student model** (`Qwen/Qwen2.5-1.5B-Instruct`) generates trajectories with logprobs
2. **Teacher model** (`Qwen/Qwen2.5-Math-7B-Instruct`) computes logprobs on the same trajectories
3. The advantage is computed as: `advantages = kl_coef * (teacher_logprobs - student_logprobs)`
4. The student model is trained to minimize this KL divergence, effectively learning from the teacher

## Key Configuration

- **Algorithm**: `on_policy_distill`
- **Workflow**: `on_policy_distill_workflow`
- **Student Model**: `Qwen/Qwen2.5-1.5B-Instruct`
- **Teacher Model**: `Qwen/Qwen2.5-Math-7B-Instruct` (configured as auxiliary model)

## Running the Example

Download the model checkpoint and modify your config file, then run:
```bash
trinity run examples/opd_gsm8k/opd_gsm8k.yaml
```

Then you are all set! It should be pretty simple😄, and the training should converge very quick.



![](../../docs/sphinx_doc/assets/opd_acc.png)
![](../../docs/sphinx_doc/assets/opd_kl.png)


## References

- https://arxiv.org/pdf/2306.13649
- https://thinkingmachines.ai/blog/on-policy-distillation/
