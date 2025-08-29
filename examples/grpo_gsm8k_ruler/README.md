# RL on GSM8K with RULER reward

This example shows a toy implementation of ART's [RULER](https://art.openpipe.ai/fundamentals/ruler) on GSM8k task and GRPO.

RULER (Relative Universal LLM-Elicited Rewards) is a general-purpose reward function that uses an LLM-as-judge to rank the rollouts for a given task.

https://github.com/OpenPipe/ART/blob/main/src/art/rewards/ruler.py


## Configurations and Metrics

The config files are located in [`gsm8k_ruler.yaml`](gsm8k_ruler.yaml) and [`train_gsm8k_ruler.yaml`](train_gsm8k_ruler.yaml).

Some key configs in this example are:
* `default_workflow_type`: set to `math_ruler_workflow`
* `auxiliary_models`: LLM-as-a-judge for RULER; need to set `max_prompt_tokens`, `max_response_tokens`, `max_model_len` appropriately
* `std_threshold` for GRPO advantage: set to small value, filter out group of experiences with same rewards (e.g., when RULER fails to return valid scores, they are set to all zero)
* `sync_style`: use `dynamic_by_explorer`, due to filtering of experiences
* `lr`: set to small value (2e-6) for stability, as rewards can be noisy


Some important metrics to pay attention to are:
* `reward`: reward calculated by RULER
* `gold_reward`: sum of `accuracy_reward` and `format_reward`, rule-based calculation with ground truth (as in original GSM8k example)
* `judge_success`: whether RULER successfully returns a valid score
* `eval_accuracy`: accuracy on the evaluation set


## Results
We show the results below:

<!-- ![](../../assets/gsm8k_ruler_reward.png) -->

Also, an example response from the judge LLM is shown below:
```text
TODO
```
