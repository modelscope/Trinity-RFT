# 常见问题

## 第一部分：参数配置
**Q:** 在哪里配置参数？

**A:** 你可以通过运行 `trinity studio --port 8080` 使用配置管理器来配置参数。这种方式提供了便捷的参数配置途径。

高级用户也可以直接编辑配置文件，参见各例子（`examples`）中的 YAML 文件。
Trinity-RFT 使用 [veRL](https://github.com/volcengine/verl) 作为训练后端，其参数数量较多，详见 [veRL 文档](https://verl.readthedocs.io/en/latest/examples/config.html)。你可以通过两种方式指定这些参数：(1) 在 `trainer.trainer_config` 字典中直接指定；(2) 在一个以 `train_` 开头的辅助 YAML 文件中指定，并将该文件路径（例如 `train_gsm8k.yaml`）传给 `trainer.trainer_config_path`。这两种方式互斥，不可同时使用。

---

**Q:** `buffer.batch_size`、`buffer.train_batch_size`、`actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu` 以及其他 batch size 参数之间有什么关系？

**A:** 这些参数的关系如下：

- `buffer.batch_size`：一个 mini-batch 中的任务（task）数量，对 explorer 有效。
- `buffer.train_batch_size`：一个 mini-batch 中的 experience 数量，对 trainer 有效。如果未显式指定，则默认为 `buffer.batch_size` * `algorithm.repeat_times`。
- `actor_rollout_ref.actor.ppo_mini_batch_size`：一个 mini-batch 中的 experience 数量，会被 `buffer.train_batch_size` 覆盖；但在 `update_policy` 函数中，其值表示每个 GPU 上的 mini-batch experience 数量，即 `buffer.train_batch_size (/ ngpus_trainer)`。除以 `ngpus_trainer` 是由于数据隐式分配到多个 GPU 上所致，但在梯度累积后不影响最终结果。
- `actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu`：每个 GPU 上 micro-batch 中的 experience 数量。

以下例子简要地展示了它们的用法：

```python
def update_policy(batch_exps):
    dataloader = batch_exps.split(ppo_mini_batch_size)
    for _ in range(ppo_epochs):
        for batch_idx, data in enumerate(dataloader):
            # 分割数据
            mini_batch = data
            if actor_rollout_ref.actor.use_dynamic_bsz:
                micro_batches, _ = rearrange_micro_batches(
                        batch=mini_batch, max_token_len=max_token_len
                    )
            else:
                micro_batches = mini_batch.split(actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu)

            # 计算梯度
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

            # 优化器更新
            grad_norm = self._optimizer_step()
    self.actor_optimizer.zero_grad()
```
详细实现请参考 `trinity/trainer/verl/dp_actor.py`。veRL 也在其 [FAQ](https://verl.readthedocs.io/en/latest/faq/faq.html#what-is-the-meaning-of-train-batch-size-mini-batch-size-and-micro-batch-size) 中对此进行了说明。

## 第二部分：常见报错

**报错：**
```bash
File ".../flash_attn/flash_attn_interface.py", line 15, in ‹module>
    import flash_attn_2_cuda as flash_attn_gpu
ImportError: ...
```

**A:** `flash-attn` 模块未正确安装。尝试运行 `pip install flash-attn==2.8.1` 或 `pip install flash-attn==2.8.1 -v --no-build-isolation` 来修复。

---

**报错：**
```bash
UsageError: api_key not configured (no-tty). call wandb.login(key=[your_api_key]) ...
```

**A:** 如果你使用 WandB 来观察实验，在启动 Ray 和运行实验之前，请先登录 WandB。一种方法是执行命令 `export WANDB_API_KEY=[your_api_key]`。你也可以选择其他方式来观察实验，比如设置 `monitor.monitor_type=tensorboard/mlflow`。

---

**报错：**
```bash
ValueError: Failed to look up actor with name 'explorer' ...
```

**A:** 确保在运行实验前已启动 Ray。如果 Ray 已在运行，可通过以下命令重启：

```bash
ray stop
ray start --head
```

---

**报错：** 内存不足 (OOM) 错误

**A:** 以下参数可能有所帮助：

- 对于 trainer：当 `actor_rollout_ref.actor.use_dynamic_bsz=false` 时，调整 `actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu`；当 `actor_rollout_ref.actor.use_dynamic_bsz=true` 时，调整 `actor_rollout_ref.actor.ppo_max_token_len_per_gpu` 和 `actor_rollout_ref.actor.ulysses_sequence_parallel_size`。设置 `actor_rollout_ref.actor.entropy_from_logits_with_chunking=true` 也可能有帮助。
- 对于 explorer：调整 `explorer.rollout_model.tensor_parallel_size`。

## 第三部分：调试方法
Trinity-RFT 现在支持 actor 级别的日志功能，可自动将每个 actor（例如 explorer 和 trainer）的日志保存到 `<checkpoint_job_dir>/log/<actor_name>` 目录下。如需查看更详细的日志信息，可通过在配置文件中设置 `log.level=debug`，将默认日志级别（`info`）更改为 `debug`。

你也可以查看所有进程的完整日志并保存到 `debug.log`：

```bash
export RAY_DEDUP_LOGS=0
trinity run --config grpo_gsm8k/gsm8k.yaml 2>&1 | tee debug.log
```

## 第四部分：其他问题
**Q:** `buffer.trainer_input.experience_buffer.path` 的作用是什么？

**A:** 该路径指定了用于持久化存储生成的 experience 的 SQLite 数据库路径。如果你不想使用 SQLite 数据库，可以注释掉这一行。

要查看数据库中的 experience，可以使用以下 Python 脚本：

```python
from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool
from trinity.common.schema.sql_schema import ExperienceModel

engine = create_engine(buffer.trainer_input.experience_buffer.path)
session = sessionmaker(bind=engine)
sess = session()

MAX_EXPERIENCES = 4
experiences = (
    sess.query(ExperienceModel)
    .limit(MAX_EXPERIENCES)
    .all()
)

exp_list = []
for exp in experiences:
    exp_list.append(ExperienceModel.to_experience(exp))

# 打印 experience 信息
for exp in exp_list:
    print(f"{exp.prompt_text=}", f"{exp.response_text=}")
```

---

**Q:** 如何在 Trinity-RFT 框架之外加载检查点（checkpoints）？

**A:** 你需要指定模型路径和检查点路径。以下代码片段展示了如何使用 transformers 库进行加载。

以下是加载 FSDP trainer 检查点的示例：

```python
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from trinity.common.models.utils import load_fsdp_state_dict_from_verl_checkpoint

# 假设我们需要第 780 步的检查点；
# model_path、checkpoint_root_dir、project 和 name 已定义
model = AutoModelForCausalLM.from_pretrained(model_path)
ckp_path = os.path.join(checkpoint_root_dir, project, name, "global_step_780", "actor")
model.load_state_dict(load_fsdp_state_dict_from_verl_checkpoint(ckp_path))
```
