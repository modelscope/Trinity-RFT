# Developer Guide

Trinity-RFT divides the RL training process into three modules: **Explorer**, **Trainer**, and **Buffer**. Each module provides extension interfaces that developers can use to implement their own modules, enabling customized development of Trinity-RFT.

The table below lists the main functions of each module, the corresponding extension interfaces, and development goals. Developers can refer to the corresponding module development tutorials and choose to extend based on their needs.

| Module   | Main Function                                | Extension Interface | Development Goal        | Tutorial Link              |
|----------|----------------------------------------------|---------------------|-------------------------|----------------------------|
| Explorer | Responsible for Agent-Environment interaction and generating trajectory data | `Workflow`          | Extend existing RL algorithms to new scenarios | [🔗](./develop_workflow.md) |
| Trainer  | Responsible for model training and updating | `Algorithm`         | Design new RL algorithms | [🔗](./develop_algorithm.md) |
| Buffer   | Responsible for storing and preprocessing task and generated trajectory data | `Operator`          | Design new data cleaning and augmentation strategies | [🔗](./develop_operator.md) |

```{tip}
Trinity-RFT provides a modular development approach, allowing you to flexibly add custom modules without modifying the framework code.
You can place your module code in the `trinity/plugins` directory. Trinity-RFT will automatically load all Python files in that directory at runtime and register the custom modules within them.
Trinity-RFT also supports specifying other directories at runtime by setting the `--plugin-dir` option, for example: `trinity run --config <config_file> --plugin-dir <your_plugin_dir>`.
```

For modules you plan to contribute to Trinity-RFT, please follow these steps:

1. Implement your code in the appropriate directory, such as `trinity/common/workflows` for `Workflow`, `trinity/algorithm` for `Algorithm`, and `trinity/buffer/operators` for `Operator`.

2. Register your module in the corresponding `__init__.py` file of the directory.

3. Add tests for your module in the `tests` directory, following the naming conventions and structure of existing tests.

4. Before submitting your code, ensure it passes the code style check by running `pre-commit run --all-files`.

5. Submit a Pull Request to the Trinity-RFT repository, providing a detailed description of your module's functionality and purpose.
