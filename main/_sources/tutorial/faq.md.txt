# FAQ

## Part 1: Configurations
**Q:** Why do most examples have two configuration YAML files, e.g., `gsm8k.yaml` and `train_gsm8k.yaml` in the `examples/grpo_gsm8k` directory?

**A:** Trinity-RFT uses [veRL](https://github.com/volcengine/verl) as the training backend, and the auxiliary YAML file starting with `train_` is used for configuring veRL, referred to [veRL documentation](https://verl.readthedocs.io/en/latest/examples/config.html).
If you specify the path to `train_gsm8k.yaml` in `trainer.trainer_config_path`, Trinity-RFT will automatically pass the parameters to veRL.

We provide an alternative way to configure the veRL trainer. You may also directly specify the parameters in the `trainer.trainer_config` dictionary. This approach is mutually exclusive with using `trainer.trainer_config_path`.

Note that some parameters are not listed in the auxiliary configuration file (e.g., `train_gsm8k.yaml`), as they will be overridden by the parameters in the trinity configuration file (e.g., `gsm8k.yaml`). Please refer to `./trinity_configs.md` for more details.
For users' convenience, future versions will gradually reduce parameters in `trainer.trainer_config` and `trainer.trainer_config_path` until it's fully deprecated.

---

**Q:** What's the relationship between `buffer.batch_size`, `actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu` and other batch sizes?

**A:** The following parameters are closely related:

- `buffer.batch_size`: The number of tasks in a batch, effective for both the explorer and the trainer.
- `actor_rollout_ref.actor.ppo_mini_batch_size`: The number of experiences in a mini-batch, overridden by `buffer.train_batch_size`; but in the `update_policy` function, its value becomes the number of experiences in a mini-batch per GPU, i.e., `buffer.train_batch_size (/ ngpus_trainer)`. The expression of dividing `ngpus_trainer` is caused by implict data allocation to GPUs, but this do not affects the result after gradient accumulation.
- `actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu`: The number of experiences in a micro-batch per GPU.

A minimal example showing their usage is as follows:

```python
def update_policy(batch_exps):
    dataloader = batch_exps.split(ppo_mini_batch_size)
    for _ in range(ppo_epochs):
        for batch_idx, data in enumerate(dataloader):
            # Split data
            mini_batch = data
            if actor_rollout_ref.actor.use_dynamic_bsz:
                micro_batches, _ = rearrange_micro_batches(
                        batch=mini_batch, max_token_len=max_token_len
                    )
            else:
                micro_batches = mini_batch.split(actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu)

            # Computing gradient
            for data in micro_batches:
                entropy, log_prob = self._forward_micro_batch(
                    micro_batch=data, ...
                )
                pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = compute_policy_loss(
                    log_prob=log_prob, **data
                )
                policy_loss = pg_loss + ...
                loss = policy_loss / self.gradient_accumulation
                loss.backward()

            # Optimizer step
            grad_norm = self._optimizer_step()
    self.actor_optimizer.zero_grad()
```
Please refer to `trinity/trainer/verl/dp_actor.py` for detailed implementation. veRL also provides an explanation in [FAQ](https://verl.readthedocs.io/en/latest/faq/faq.html#what-is-the-meaning-of-train-batch-size-mini-batch-size-and-micro-batch-size).


## Part 2: Common Errors

**Error:**
```bash
File ".../flash_attn/flash_attn_interface.py", line 15, in ‹module>
    import flash_attn_2_cuda as flash_attn_gpu
ImportError: ...
```

**A:** The `flash-attn` module is not properly installed. Try to fix it by running `pip install flash-attn==2.8.0.post2` or `pip install flash-attn==2.8.0.post2 -v --no-build-isolation`.

---

**Error:**
```bash
UsageError: api_key not configured (no-tty). call wandb.login(key=[your_api_key]) ...
```

**A:** Try to log in to WandB before starting Ray and running the experiment. One way to do this is run the command `export WANDB_API_KEY=[your_api_key]`.

---

**Error:**
```bash
ValueError: Failed to look up actor with name 'explorer' ...
```

**A:** Make sure Ray is started before running the experiment. If Ray is already running, you can restart it with the following commands:

```bash
ray stop
ray start --head
```

---

**Error:** Out-of-Memory (OOM) error

**A:** The following parameters may be helpful:

- For trainer, adjust `actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu` when `actor_rollout_ref.actor.use_dynamic_bsz=false`; adjust `actor_rollout_ref.actor.ppo_max_token_len_per_gpu` and `actor_rollout_ref.actor.ulysses_sequence_parallel_size` when `actor_rollout_ref.actor.use_dynamic_bsz=true`. Setting `actor_rollout_ref.actor.entropy_from_logits_with_chunking=true` may also help.
- For explorer, adjust `explorer.rollout_model.tensor_parallel_size`,


## Part 3: Debugging Methods [Coming Soon]
To see the full logs of all processes and save it to `debug.log`:
```bash
export RAY_DEDUP_LOGS=0
trinity run --config grpo_gsm8k/gsm8k.yaml 2>&1 | tee debug.log
```


## Part 4: Other Questions
**Q:** What's the purpose of `buffer.trainer_input.experience_buffer.path`?

**A:** This path specifies the path to the SQLite database storaging the generated experiences. You may comment out this line if you don't want to use the SQLite database.

To see the experiences in the database, you can use the following Python script:

```python
from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool
from trinity.common.schema import ExperienceModel

engine = create_engine(buffer.trainer_input.experience_buffer.path)
session = sessionmaker(bind=engine)
sess = session()

MAX_EXPERIENCES = 4
experiences = (
    sess.query(ExperienceModel)
    .with_for_update()
    .limit(MAX_EXPERIENCES)
    .all()
)

exp_list = []
for exp in experiences:
    exp_list.append(ExperienceModel.to_experience(exp))

# Print the experiences
for exp in exp_list:
    print(f"{exp.prompt_text=}", f"{exp.response_text=}")
```

---

**Q:** How to load the checkpoints outside of the Trinity-RFT framework?

**A:** You need to specify model path and checkpoint path. The following code snippet gives an example with transformers.

```python
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from trinity.common.models.utils import load_state_dict_from_verl_checkpoint

# Assume we need the checkpoint at step 780;
# model_path, checkpoint_root_dir, project, and name are already defined
model = AutoModelForCausalLM.from_pretrained(model_path)
ckp_path = os.path.join(checkpoint_root_dir, project, name, "global_step_780", "actor")
model.load_state_dict(load_state_dict_from_verl_checkpoint(ckp_path))
```
