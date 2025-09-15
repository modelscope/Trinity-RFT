# 开发者指南

本指南介绍了如何在 Trinity-RFT 中开发新模块，并提供相关的开发规范。

在 Trinity-RFT 中，我们将 RL 流程分解为三个主要模块（**Explorer**、**Trainer** 和 **Buffer**），以便于自定义和扩展。
下表总结了不同开发目标的开发者需要关注的核心模块和关键组件。

| 开发目标 | 核心模块 | 关键组件 |
|--------------------|-------------|---------------|
| 将现有 RL 算法应用于新环境 | *Explorer* | `Workflow` |
| 设计新的 RL 算法 | *Trainer* | `Algorithm` |
| 从数据角度增强 RL 流程 | *Buffer* | `Operator` |

```{note}
Trinity-RFT 正在积极开发中，以下接口可能会发生变化。使用本指南时请参考最新代码。
```

```{tip}
对于仅用于本地测试或不打算贡献的模块，你可以将其放在 `trinity/plugins` 目录中。Trinity-RFT 会自动加载该目录下的所有模块，且无需将其添加到 `__init__.py` 文件中。你也可以通过运行 Trinity-RFT 时设置 `--plugin-dir` 选项来指定其他目录，例如：`trinity run --config /path/to/your/config --plugin-dir /path/to/your/plugins`。
```

---

(Workflows)=
## 工作流（适用于 RL 环境开发者）

在 Trinity-RFT 中，工作流（Workflow）是定义 Agent 与 Environment 之间交互的核心组件。
一个合格的工作流需要使用训练好的模型完成指定任务，并从环境中获取反馈信息（奖励）。以下是创建新工作流的步骤：

---

### 步骤 0：基本概念

在开始开发之前，理解以下几个核心概念非常重要：

- **任务（Task）** ({class}`trinity.common.workflows.Task`)：表示可转换为 `Workflow` 的数据结构。`Task` 的内容根据任务类型而异：
  - **数学问题**：`Task` 包含问题描述和标准答案。
  - **编程场景**：`Task` 包括问题描述、测试用例、运行环境等复杂信息。

- **工作流（Workflow）** ({class}`trinity.common.workflows.Workflow`)：描述 `Task` 如何执行。它定义了 Agent 与 Environment 的交互流程，包括类似其他框架中的 *Rollout* 和 *Reward* 计算逻辑。执行后生成一组 `Experience`。Trinity-RFT 包含多个内置工作流：
  - `MathWorkflow` ({class}`trinity.common.workflows.MathWorkflow`)：用于数学场景，将问题提交给 LLM，解析 LLM 响应，并计算分数（奖励）。
  - `WebShopWorkflow` ({class}`trinity.common.workflows.WebShopWorkflow`)：用于 webshop 场景，包含与环境的多轮交互。
  - `CodeWorkflow`（即将推出）：用于编码场景，执行返回的代码，运行测试，并根据测试结果计算奖励。
  - ...

- **经验（Experience）** ({class}`trinity.common.experience.Experience`)：运行 `Workflow` 的输出。内部数据格式取决于所使用的训练算法。例如，对于常见的 PPO/GRPO 算法，`Experience` 包含 token ID 列表、动作掩码（标识哪些 token 是由 LLM 生成的）、对数概率、奖励等。

---

### 步骤 1：准备任务数据集

任务数据集通过 YAML 配置文件中的 `buffer.explorer_input.taskset` 配置项加载。
为处理 `Task` 内容的差异，Trinity-RFT 提供了一个统一的 `Task` 接口，包含以下字段：

- **`workflow`** (`str`)：你的工作流类的注册名称。你可以在 YAML 配置文件的 `buffer.explorer_input.taskset.default_workflow_type` 中指定。
- **`reward_fn`** (`Optional[str]`)：你的奖励函数的注册名称。你可以在 `buffer.explorer_input.taskset.default_reward_fn_type` 中指定。注意某些工作流已内置奖励计算；此时可省略该字段。
- **`raw_task`** (`Dict`)：原始数据的记录，以 `Dict` 格式存储。对于高度定制化的工作流，你可以直接使用 `raw_task` 初始化 `Workflow` 实例，而不依赖以下字段。
- **`format_args`** ({class}`trinity.common.config.FormatConfig`)：便于构造 `Workflow` 实例的参数。例如，`prompt_key` 和 `response_key` 可用于从 `raw_task` 中提取 prompt 和 response。这些设置来自 YAML 配置文件，可在 `buffer.explorer_input.task_set.format` 中设置。
- **`rollout_args`** ({class}`trinity.common.config.GenerationConfig`)：控制 rollout 过程的参数，如 `temperature`。该字段也来自 YAML 配置文件，可在 `buffer.explorer_input.task_set.rollout_args` 中设置。
- **`workflow_args`** (`Dict`)：用于构造 `Workflow` 实例的参数字典。相比 `format_args` 和 `rollout_args` 更灵活。该字段也来自 YAML 配置文件，可在 `buffer.explorer_input.task_set.workflow_args` 中设置。通常无需设置此字段。

```{tip}
`workflow`、`workflow_args` 和 `raw_task` 提供了不同级别的自定义能力。

- `workflow` 为使用相同工作流的所有任务提供全局设置。（全局级别）
- `workflow_args` 可为每个任务数据集设置，允许使用相同工作流的不同任务数据集表现出不同行为。（数据集级别）
- `raw_task` 提供对每个任务行为的自定义能力，最为灵活。（数据样本级别）
```

在数学问题场景中，`Task` 数据集可以是一个 `jsonl` 文件，每行包含带有 `question` 和 `answer` 字段的 JSON，分别表示问题描述和标准答案。例如：

```json
{"question": "1+1=", "answer": "2"}
{"question": "2+2=", "answer": "4"}
...
```

配置示例片段：

```yaml
# some config
buffer:
  explorer_input:
    taskset:
      default_workflow: "math_workflow"
      path: ${oc.env:TRINITY_TASKSET_PATH}
      format:
        prompt_key: "question"
        response_key: "answer"
      rollout_args:
        temperature: 1.0
      # some other configs
```

在此示例中，每个任务对象的 `raw_task` 是一个包含两个键（`question` 和 `answer`）的 `Dict`。`MathWorkflow` 使用 `prompt_key` 和 `response_key` 从 `raw_task` 中提取问题和答案，并使用 `rollout_args` 生成响应。

---

### 步骤 2：实现新的工作流

`Workflow` 基类接口如下：

```python
class Workflow(ABC):

    def __init__(
        self,
        *,
        task: Task,
        model: ModelWrapper,
        auxiliary_models: Optional[List[openai.OpenAI]] = None,
    ):
        self.task = task
        self.model = model
        self.auxiliary_models = auxiliary_models

    @abstractmethod
    def run(self) -> List[Experience]:
        """Run the workflow and return a list of Experiences."""
```

#### 初始化你的工作流

`Workflow` 接受以下初始化参数：

- `task`({class}`trinity.common.workflows.Task`)：数据集中的单个任务。
- `model`({class}`trinity.common.models.model.ModelWrapper`)：正在训练的模型，提供类似于 OpenAI 的接口，能够接收对话消息列表并返回 LLM 生成的内容（包括回复文本 `response_text`、完整序列 token id `tokens`、prompt 部分 token 长度 `prompt_length`，以及输出 token 对数概率列表 `logprobs`）。
- `auxiliary_models`(`List[openai.OpenAI]`)：未参与训练的辅助模型列表。所有模型均通过兼容 OpenAI 的 API 提供。

```{tip}
你可以在配置文件中将 `explorer.rollout_model.enable_openai_api` 设置为 `true`，并在工作流中调用 `model.get_openai_client()` 获取 `openai.OpenAI` 实例，从而切换为使用 OpenAI API。
调用 OpenAI API 时，`model` 字段可通过 `openai_client.models.list().data[0].id` 或 `openai_client.model_path` 获取。
```

以下是一个仅使用 `raw_task` 和 `rollout_args` 初始化简单工作流的示例。在更复杂的情况下，你可以使用 `format_args` 进行进一步自定义。

```python
class ExampleWorkflow(Workflow):

    def __init__(self, task: Task, model: ModelWrapper, auxiliary_models: List):
        super().__init__(task=task, model=model, auxiliary_models=auxiliary_models)
        self.question = task.raw_task.get("question")
        self.answer = task.raw_task.get("answer")
        self.rollout_args = task.rollout_args
        # Optional: If you want to use OpenAI API in your workflow
        # self.openai_client = self.model.get_openai_client()
```

#### 实现 `run` 方法

`run` 方法是工作流的核心方法。它返回一个 `Experience` 列表。
以下是一个数学工作流的简单实现。

我们首先调用模型，使用给定的问题和 rollout 参数生成多个响应。
然后使用 `calculate_reward` 函数计算每个响应的奖励。
最后，我们构建一个包含响应和奖励的 `Experience` 列表并返回。

```python
class ExampleWorkflow(Workflow):

    # the __init__ function

    def calculate_reward(self, response: str, truth: str) -> float:
        if response == truth:
            return 1.0
        else:
            return 0.0

    def run(self) -> List[Experience]:
        # call the model to generate multiple responses
        responses = self.model.chat(
            [
                {
                    "role": "user",
                    "content": f"Question:\n{self.question}",
                }
            ],
            n=self.rollout_args.n,
            temperature=self.rollout_args.temperature,
        )
        experiences = []
        for response in responses:
            # calulcate reward
            reward: float = self.calculate_reward(response.response_text, self.answer)
            # construct Experience
            experiences.append(
                Experience(
                    tokens=response.tokens,
                    prompt_length=response.prompt_length,
                    reward=reward,
                    logprobs=response.logprobs,
                )
            )
        return experiences
```

#### 注册你的工作流

使用 `WORKFLOWS.register_module` 装饰器注册你的工作流。
确保名称不与现有工作流冲突。

```python
# import some packages
from trinity.common.workflows.workflow import WORKFLOWS

@WORKFLOWS.register_module("example_workflow")
class ExampleWorkflow(Workflow):
    pass
```

对于准备贡献给 Trinity-RFT 项目的模块，你需要将上述代码放入 `trinity/common/workflows` 文件夹中，例如 `trinity/common/workflows/example_workflow.py`。并在 `trinity/common/workflows/__init__.py` 中添加以下行：

```python
# existing import lines
from .example_workflow import ExampleWorkflow

__all__ = [
    # existing __all__ lines
    "ExampleWorkflow",
]
```

#### 避免重复初始化

对于重型工作流，每次重新初始化会带来额外计算开销。
此时，你可以实现 `resettable` 和 `reset` 方法以避免重复初始化。

```python
@WORKFLOWS.register_module("example_workflow")
class ExampleWorkflow(Workflow):
    # some code
    # ...

    def resettable(self):
        return True

    def reset(self, task: Task):
        self.question = task.raw_task.get("question")
        self.answer = task.raw_task.get("answer")
```

#### 完整代码示例

```python
@WORKFLOWS.register_module("example_workflow")
class ExampleWorkflow(Workflow):

    def __init__(self, task: Task, model: ModelWrapper, auxiliary_models: List):
        super().__init__(task=task, model=model, auxiliary_models=auxiliary_models)
        self.question = task.raw_task.get("question")
        self.answer = task.raw_task.get("answer")
        self.rollout_args = task.rollout_args

    def calculate_reward(self, response: str, truth: str) -> float:
        if response == truth:
            return 1.0
        else:
            return 0.0

    def run(self) -> List[Experience]:
        # call the model to generate multiple responses
        responses = self.model.chat(
            [
                {
                    "role": "user",
                    "content": f"Question:\n{self.question}",
                }
            ],
            n=self.rollout_args.n,
            temperature=self.rollout_args.temperature,
        )
        experiences = []
        for response in responses:
            # calulcate reward
            reward: float = self.calculate_reward(response.response_text, self.answer)
            # construct Experience
            experiences.append(
                Experience(
                    tokens=response.tokens,
                    prompt_length=response.prompt_length,
                    reward=reward,
                    logprobs=response.logprobs,
                )
            )
        return experiences

    def resettable(self):
        return True

    def reset(self, task: Task):
        self.question = task.raw_task.get("question")
        self.answer = task.raw_task.get("answer")
```

---

### 步骤 3：使用你的工作流

实现并注册工作流后，你需要更新配置文件，将 `buffer.explorer_input.taskset` 域中的 `default_workflow_type` 设置为新注册的 `Workflow` 名称。

```yaml
buffer:
  # Other fields
  explorer_input:
    taskset:
      path: /path/to/taskset
      default_workflow_type: example_workflow
      # Other fields
```

现在你可以使用以下命令在 Trinity-RFT 中运行你的工作流：

```
trinity run --config <your_yaml_file>
```

---

(Algorithms)=
## 算法（适用于 RL 算法开发者）

Trinity-RFT 提供了实现新算法的标准化流程。

### 步骤 0：算法模块基本概念

在 Trinity-RFT 中，算法模块主要负责在 RL 过程中从回放缓冲区提取 experience 数据，并基于这些数据计算损失以更新模型。
为了避免每次添加新算法时都实现新的 Trainer 类，我们将典型的 PPO 算法流程分解为多个子模块，以适应各种算法。

- **采样策略（Sample Strategy）** ({class}`trinity.algorithm.SampleStrategy`)：负责从缓冲区模块中采样 experience 数据。通过自定义此模块，你可以实现过滤 experience 数据或从多个数据源混合采样的功能。
- **优势函数（Advantage Fn）**({class}`trinity.algorithm.AdvantageFn`)：负责计算 experience 数据的优势值（Advantage）和回报值（Returns）。
- **策略损失函数（Policy Loss Fn）**({class}`trinity.algorithm.PolicyLossFn`)：负责计算策略网络的核心训练损失。
- **KL 函数（KL Fn）**({class}`trinity.algorithm.KLFn`)：负责计算 KL 散度，通常在现有 RL 算法中用于两个地方：奖励惩罚和 Actor 损失。
- **熵损失函数（Entropy Loss Fn）**({class}`trinity.algorithm.EntropyLossFn`)：负责计算策略网络的熵损失。

我们在 `trinity/algorithm` 中提供了上述模块的若干实现。

---

### 步骤 1：实现算法组件

Trinity-RFT 允许开发者自定义所有上述模块。开发者只需根据新算法的需求实现特定模块。本节将以 {ref}`OPMD <OPMD>` 算法为例进行简要介绍。

OPMD 与 PPO 算法的主要区别在于优势值和策略损失的计算。
OPMD 依赖于基于组的优势值计算，且不使用 Critic 模型。
要实现 OPMD，开发者需要在 `AdvantageFn` 中实现优势值计算，在 `PolicyLossFn` 中实现策略损失计算。

---

#### 步骤 1.1：实现 `AdvantageFn`

{class}`trinity.algorithm.AdvantageFn` 接口包含三个方法：

- `__call__`：优势值计算的主要入口。接收一个 experience 列表 ({class}`trinity.common.experience.Experience`)，返回一个包含计算出的优势值和回报值的 experience 列表，以及一个用于日志记录的指标字典。
- `default_args`：类方法，返回默认初始化参数（字典形式）。当用户未在配置文件中指定初始化参数时，默认使用此方法返回的参数。
- `compute_in_trainer`：类方法，指示是否在 Trainer 中计算优势值。若返回 `False`，则 `AdvantageFn` 将在 experience 数据处理流水线中被调用。

为方便起见，Trinity-RFT 提供了一个抽象类 {class}`trinity.algorithm.advantage_fn.GroupAdvantage`，它实现了基于组的优势值计算的 `__call__` 方法，你可以专注于如何对 experience 进行分组以及如何在分组后的 experience 上计算优势值，通过以下两个方法实现：

- `group_experiences`：此方法将一步生成的 experience 划分为多个子组。

- `calculate_group_advantage`：此方法计算每组 experience 的优势值。

以下是 OPMD 算法优势函数的实现示例：

```python
from trinity.algorithm.advantage_fn import ADVANTAGE_FN, GroupAdvantage

@ADVANTAGE_FN.register_module("opmd")
class OPMDGroupAdvantage(GroupAdvantage):
    """OPMD Group Advantage computation"""

    def __init__(self, opmd_baseline: str = "mean", tau: float = 1.0, **kwargs) -> None:
        super().__init__(**kwargs)
        self.opmd_baseline = opmd_baseline
        self.tau = tau

    def group_experiences(self, exps):
        return group_by(exps, id_type="task")

    def calculate_group_advantage(
        self, group_id: str, exps: List[Experience]
    ) -> Tuple[List[Experience], Dict]:
        with torch.no_grad():
            if len(exps) == 1:
                group_baseline = torch.tensor(0.0)
            else:
                group_rewards = torch.tensor([exp.reward for exp in exps], dtype=torch.float32)
                if self.opmd_baseline == "mean":
                    group_baseline = torch.mean(group_rewards)
                else:
                    group_baseline = self.tau * (
                        torch.logsumexp(group_rewards / self.tau, dim=-1)
                        - torch.log(torch.tensor(len(exps)))
                    )
            for exp in exps:
                score = exp.reward - group_baseline
                exp.advantages = score * exp.action_mask
                exp.returns = exp.advantages.clone()
            metrics = {
                "group_baseline": group_baseline.item(),
            }
        return exps, metrics

    @classmethod
    def default_args(cls) -> dict:
        return {"opmd_baseline": "mean", "tau": 1.0}
```

实现后，你需要通过 {class}`trinity.algorithm.ADVANTAGE_FN` 注册此模块。注册后，该模块可在配置文件中使用注册名称进行配置。

#### 步骤 1.3：实现 `PolicyLossFn`

开发者需要实现 {class}`trinity.algorithm.PolicyLossFn` 接口，其与 `AdvantageFn` 类似，包含两个方法：

- `__call__`：根据输入参数计算损失。与 `AdvantageFn` 不同，这里的输入参数均为 `torch.Tensor`。该接口会自动扫描 `__call__` 方法的参数列表，并将其转换为 experience 数据中的对应字段。因此，请直接在参数列表中写出损失计算所需的所有张量名称，而不是从 `kwargs` 中选择参数。
- `default_args`：返回默认初始化参数（字典形式），当用户未在配置文件中指定初始化参数时，默认使用此方法返回的参数。

同样，实现后需要通过 {class}`trinity.algorithm.POLICY_LOSS_FN` 注册此模块。

以下是 OPMD 算法策略损失函数的实现示例。由于 OPMD 的策略损失仅需 logprob、action_mask 和 advantages，因此 `__call__` 方法的参数列表中仅指定这三个项：

```python
@POLICY_LOSS_FN.register_module("opmd")
class OPMDPolicyLossFn(PolicyLossFn):
    def __init__(self, tau: float = 1.0) -> None:
        self.tau = tau

    def __call__(  # type: ignore
        self,
        logprob: torch.Tensor,
        action_mask: torch.Tensor,
        advantages: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict]:
        pg_losses = -advantages * logprob
        opmd_loss = masked_mean(pg_losses, action_mask)
        opmd_loss = opmd_loss / (1.0 + self.tau)  # for regularization (w.r.t. current pi_theta)
        return opmd_loss, {"opmd_loss": opmd_loss.detach().item()}

    @classmethod
    def default_args(cls) -> Dict:
        return {"tau": 1.0}
```

---

### 步骤 2：注册你的算法

上述步骤实现了算法所需的组件，但这些组件是分散的，需要在多个地方配置才能生效。

为简化配置，Trinity-RFT 提供了 {class}`trinity.algorithm.AlgorithmType` 来描述完整算法，并在 {class}`trinity.algorithm.ALGORITHM_TYPE` 中注册，实现一键配置。

`AlgorithmType` 类包含以下属性和方法：

- `use_critic`：是否使用 Critic 模型
- `use_reference`：是否使用 Reference 模型
- `compute_advantage_in_trainer`：是否在 Trainer 中计算优势值；若为 False，则跳过 Trainer 中的 `AdvantageFn` 调用
- `can_balance_batch`：算法是否允许在将批次拆分为微批次时自动平衡（打乱样本顺序）
- `schema`：算法对应的 experience 数据格式
- `default_config`：获取算法的默认配置，将覆盖 `ALGORITHM_TYPE` 中同名属性

同样，实现后需要通过 `ALGORITHM_TYPE` 注册此模块。

以下是 OPMD 算法的实现。
由于 OPMD 算法不需要使用 Critic 模型，`use_critic` 设置为 `False`。
`default_config` 方法返回的字典表明 OPMD 将使用步骤 1 中实现的 `opmd` 类型的 `AdvantageFn` 和 `PolicyLossFn`，不会对奖励应用 KL 惩罚，但在计算最终损失时会添加 `k2` 类型的 KL 损失。

```python
@ALGORITHM_TYPE.register_module("opmd")
class OPMDAlgorithm(AlgorithmType):
    """OPMD algorithm."""

    use_critic: bool = False
    use_reference: bool = True
    compute_advantage_in_trainer: bool = False
    can_balance_batch: bool = True
    schema: str = "experience"

    @classmethod
    def default_config(cls) -> Dict:
        return {
            "repeat_times": 2,
            "advantage_fn": "opmd",
            "sample_strategy": "warmup",
            "policy_loss_fn": "opmd",
            "kl_penalty_fn": "none",
            "kl_loss_fn": "k2",
            "entropy_loss_fn": "default",
        }
```

---

### 步骤 3：使用你的算法

完成上述所有步骤后，你可以通过 YAML 配置文件使用新注册的算法。

对于默认配置，你只需在 `config.yaml` 文件中添加以下内容：

```yaml
# some other configs
algorithm:
  algorithm_type: "opmd"
# some other configs
```

如果需要修改某些参数，可以在 `algorithm` 部分直接添加对应参数。例如，若需要修改 `repeat_times` 以及 `AdvantageFn` 和 `PolicyLossFn` 的初始化参数，修改后的 `config.yaml` 文件如下：

```yaml
# some other configs
algorithm:
  algorithm_type: "opmd"
  repeat_times: 8
  advantage_fn_args:
    opmd_baseline: "logavgexp"
    tau: 0.99
  policy_loss_fn_args:
    tau: 0.99
# some other configs
```

---

## 数据处理算子（适用于数据开发者）

### 步骤 0：Operator 模块基本概念

在 Trinity-RFT 中，Operator 模块负责处理缓冲区模块中的 experience 数据。它天然支持来自 [Data-Juicer](https://github.com/modelscope/data-juicer) 的现有数据处理功能，也允许开发者实现自己的算子。
通过自定义数据处理算子，开发者可以实现各种数据处理功能，如数据增强、过滤和转换。你甚至可以将优势值/回报值计算实现为 Operator，如 {ref}`算法 <Algorithms>` 部分所示。

- **DataJuicerOperator** ({class}`trinity.data.operators.DataJuicerOperator`)：封装来自 Data-Juicer 的数据处理算子。为开发者提供简单接口，列出想要使用的 Data-Juicer 算子。完整的 Data-Juicer 算子列表请见 [此处](https://modelscope.github.io/data-juicer/en/main/docs/Operators.html)。
- **ExperienceOperator** ({class}`trinity.data.operators.ExperienceOperator`)：用于 experience 数据处理的所有数据处理算子的基类。定义了所有数据处理算子应具备的接口和通用功能。每个算子处理一批 experience 数据，并返回处理后的数据及用于日志记录的指标。
- **ExperiencePipeline** ({class}`trinity.data.pipelines.ExperiencePipeline`)：管理一系列数据处理算子的 experience 数据处理流水线。它从 `Explorer` 获取原始 experience，通过流水线中的每个算子处理，最后将最终处理过的 experience 写入 `Trainer` 的输入缓冲区。

```{note}
除了 `ExperiencePipeline`，Trinity-RFT 还提供 `TaskPipeline` 用于任务数据处理。
当前版本中，`TaskPipeline` 仅支持使用 Data-Juicer 算子。详情请参见 {ref}`数据处理 <Data Processing>` 部分。
```
---

开发者可通过以下步骤实现并使用自己的算子。

### 步骤 1：实现数据处理算子

`ExperienceOperator` 接口仅包含一个 `process` 方法。`ExperiencePipeline` 将调用此方法，传入 `Explorer` 在一次探索步骤中生成的一组 `Experience`。`process` 方法应返回一个元组，包含处理后的 `Experience` 列表和用于日志记录的指标字典。

```python
class ExperienceOperator(ABC):

    @abstractmethod
    def process(self, exps: List[Experience]) -> Tuple[List[Experience], Dict]:
        """Process a list of experiences and return a transformed list.

        Args:
            exps (List[Experience]): List of experiences to process, which contains
                all experiences generated by the Explorer in one explore step.
        Returns:
            Tuple[List[Experience], Dict]: A tuple containing the processed list of experiences and a dictionary of metrics.
        """
```

以下是一个简单数据处理算子的实现示例，该算子过滤掉奖励低于某一阈值的 experience：

```python
from trinity.buffer.operators import EXPERIENCE_OPERATORS, ExperienceOperator
from trinity.common.experience import Experience


@EXPERIENCE_OPERATORS.register_module("reward_filter")
class RewardFilter(ExperienceOperator):

    def __init__(self, threshold: float = 0.0) -> None:
        self.threshold = threshold

    def process(self, exps: List[Experience]) -> Tuple[List[Experience], Dict]:
        filtered_exps = [exp for exp in exps if exp.reward >= self.threshold]
        metrics = {"filtered_count": len(exps) - len(filtered_exps)}
        return filtered_exps, metrics
```

实现后，你需要通过 {class}`trinity.data.operators.EXPERIENCE_OPERATORS` 注册此模块。注册后，该模块可在配置文件中使用注册名称进行配置。

### 步骤 2：使用此算子

完成上述步骤后，你可以通过 YAML 配置文件使用新注册的算子。

```yaml
# some other configs
data_processor:
  experience_pipeline:
    operators:
      - name: "reward_filter"
        args:
          threshold: 0.1
synchronizer:
  sync_method: nccl
  sync_style: dynamic_by_explorer
  sync_interval: 2
# some other configs
```

```{tip}
`RewardFilter` 会减少 experience 数量，可能导致 Trainer 无法获得足够的 experience 启动训练步骤。为避免此问题，你可以使用 Trinity-RFT 提供的高级 {ref}`动态同步 <Synchronizer>` 功能，如上配置所示。
上述设置意味着 `Explorer` 每 2 步与 `Trainer` 同步一次，且无论 `Trainer` 完成了多少步都会继续运行。这确保了只要 `Explorer` 在运行，`Trainer` 就总能获得足够的 experience 来启动训练步骤。
```

---

## 为配置生成器添加新配置项（高级）

### 步骤 0：了解 Streamlit

在为配置生成器页面添加新参数之前，必须熟悉 [Streamlit](https://docs.streamlit.io/develop/api-reference) 的相关 API 和机制。本项目主要使用 Streamlit 的各种输入组件，并利用 `st.session_state` 存储用户输入的参数。

### 步骤 1：实现新配置项

为了说明如何为配置生成器页面创建新参数设置，我们以 `train_batch_size` 为例。

1. 确定参数的合适作用域。目前参数分为四个文件：
   - `trinity/manager/config_registry/buffer_config_manager.py`
   - `trinity/manager/config_registry/explorer_config_manager.py`
   - `trinity/manager/config_registry/model_config_manager.py`
   - `trinity/manager/config_registry/trainer_config_manager.py`

   本例中，`train_batch_size` 应放在 `buffer_config_manager.py` 文件中。

2. 使用 Streamlit 创建参数设置函数。函数名必须以 'set_' 开头，其余部分成为配置名称。

3. 使用 `CONFIG_GENERATORS.register_config` 装饰器装饰参数设置函数。该装饰器需要以下信息：
   - 参数的默认值
   - 可见性条件（如适用）
   - 额外配置参数（如需要）

```{note}
`CONFIG_GENERATORS.register_config` 装饰器会自动将 `key=config_name` 作为参数传递给注册的配置函数。确保你的函数接受此关键字参数。
```

对于 `train_batch_size`，我们将使用以下设置：

- 默认值：96
- 可见性条件：`lambda: st.session_state["trainer_gpu_num"] > 0`
- 额外配置：`{"_train_batch_size_per_gpu": 16}`

以下是 `train_batch_size` 参数的完整代码：

```python
@CONFIG_GENERATORS.register_config(
    default_value=96,
    visible=lambda: st.session_state["trainer_gpu_num"] > 0,
    other_configs={"_train_batch_size_per_gpu": 16},
)
def set_train_batch_size(**kwargs):
    key = kwargs.get("key")
    trainer_gpu_num = st.session_state["trainer_gpu_num"]
    st.session_state[key] = (
        st.session_state["_train_batch_size_per_gpu"] * st.session_state["trainer_gpu_num"]
    )

    def on_change():
        st.session_state["_train_batch_size_per_gpu"] = max(
            st.session_state[key] // st.session_state["trainer_gpu_num"], 1
        )

    st.number_input(
        "Train Batch Size",
        min_value=trainer_gpu_num,
        step=trainer_gpu_num,
        help=_str_for_train_batch_size(),
        on_change=on_change,
        **kwargs,
    )
```

如果参数需要验证，创建一个检查函数。对于 `train_batch_size`，我们需要确保它能被 `trainer_gpu_num` 整除。若不能，则显示警告，并将参数添加到 `unfinished_fields`。

使用 `CONFIG_GENERATORS.register_check` 装饰器装饰检查函数：

```python
@CONFIG_GENERATORS.register_check()
def check_train_batch_size(unfinished_fields: set, key: str):
    if st.session_state[key] % st.session_state["trainer_gpu_num"] != 0:
        unfinished_fields.add(key)
        st.warning(_str_for_train_batch_size())
```

```{note}
`CONFIG_GENERATORS.register_check` 装饰器会自动接收 `key=config_name` 和 `unfinished_fields=self.unfinished_fields` 作为参数。确保你的函数接受这些关键字参数。
```

### 步骤 2：将新参数集成到 `config_manager.py`

要成功将新参数集成到 `config_manager.py` 文件中，请遵循以下步骤：

1. **参数分类**：
   根据其功能确定新参数的合适部分。配置生成器页面分为两种主要模式：
   - 初学者模式：包含“基本配置”和“重要配置”部分。
   - 专家模式：包含“模型”、“缓冲区”、“Explorer 和 Synchronizer”以及“Trainer”部分。

2. **添加参数**：
   在 `ConfigManager` 类的 `self.get_configs` 方法中，将新参数添加到相应部分。

   示例：

   ```python
   class ConfigManager:
       def _expert_buffer_part(self):
           self.get_configs("total_epochs", "train_batch_size")
   ```

3. **集成到 YAML 文件**：
   在 YAML 文件结构中找到新参数的合适位置。应在 `generate_config` 函数及其关联子函数中完成。

4. **赋值参数值**：
   使用 `st.session_state` 从配置生成器页面获取参数值，并将其赋给 YAML 中的对应字段。

   示例：

   ```python
   class ConfigManager:
       def _gen_buffer_config(self):
           buffer_config = {
               "batch_size": st.session_state["train_batch_size"],
               # Additional configuration parameters
           }
   ```

严格遵循这些步骤，你可以确保新参数成功添加到配置生成器页面，并正确集成到配置系统中。此过程保持了配置管理框架的完整性和功能性。

---

## 贡献你的代码

对于准备贡献给 Trinity-RFT 项目的模块，请遵循以下步骤：

1. 在适当目录中实现你的代码，例如 `trinity/common/workflows` 用于工作流，`trinity/algorithm` 用于算法，`trinity/buffer/operators` 用于操作符。

2. 在目录对应的 `__init__.py` 文件中注册你的模块。

3. 在 `tests` 目录中为你的模块添加测试，遵循现有测试的命名约定和结构。

4. 提交代码前，确保通过 `pre-commit run --all-files` 完成代码风格检查。

5. 向 Trinity-RFT 仓库提交 Pull Request，包含对你更改的清晰描述。
