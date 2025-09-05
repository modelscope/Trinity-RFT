# Train one model for both policy and reward


Ref: ART's RULER; Kimi-k2.


Simulate a scenario where only a fraction (`PROBABILITY_GROUND_TRUTH_AVAILABLE = 0.2`) of tasks have ground-truth answers.
Two RL objectives are optimized jointly: one for solution generation, the other for RULER-reward generation.


## Configurations and Metrics

The config files are located in [`gsm8k_ruler.yaml`](gsm8k_ruler.yaml) and [`train_gsm8k_trainable_ruler.yaml`](train_gsm8k_trainable_ruler.yaml).

Some key configs in this example are:

* `default_workflow_type`: set to `math_trainable_ruler_workflow`
* `std_threshold` for GRPO advantage: set to small value, filter out group of experiences with same rewards (e.g., when RULER fails to return valid scores, they are set to all zero)
* `sync_style`: use `dynamic_by_explorer`, due to filtering of experiences
* `train_batch_size`: set to 960; note that one explore step can generate more than 96 * 8 = 768 experiences
* `lr`: set to small value (2e-6) for stability, as rewards can be noisy



Some important metrics to pay attention to are:

* `reward`: reward calculated by rule or by RULER
* `gold_reward`: sum of `accuracy_reward` and `format_reward`, rule-based calculation with ground truth
* `judge_success`: whether RULER successfully returns a valid score (a coarse estimation, mix up two types of experiences)
* `reward_for_judger`: reward for the LLM working as a RULER reward model, calculated by mean absolute error (MAE) distance from gold scores
* `eval_accuracy`: accuracy on the evaluation set (ultimate metric for success of RL)


## Results

(TODO)

Compare with baseline: previous RULER workflow with Qwen2.5-1.5B-Instruct as LLM judge (`auxiliary_models`)



## Potential improvements

balance number of samples / loss weights for generation vs for RULER
