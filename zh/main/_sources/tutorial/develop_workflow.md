(Workflows)=
## Workflow 开发指南

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
  -  `AgentScopeReactMathWorkflow` ({class}`trinity.common.workflows.AgentScopeReactMathWorkflow`)：用于数学场景，直接使用现有的 ReActAgent（基于 AgentScope）来解决数学问题。

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
