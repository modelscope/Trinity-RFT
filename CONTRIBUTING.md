# Contributing to Trinity-RFT

Thank you for your interest in Trinity-RFT! Our framework is built on a decoupled architecture consisting of the **Explorer**, **Trainer**, and **Buffer**. We welcome all forms of contributions—from core feature enhancements and new algorithms to documentation and bug reports.

## Where to Contribute

Trinity-RFT provides modular interfaces for different technical interests. Please refer to our [Developer Guide](https://modelscope.github.io/Trinity-RFT/en/main/tutorial/develop_overview.html) for detailed implementation standards:

| Focus Area | Interface/Code Directory | Potential Tasks |
| :--- | :--- | :--- |
| **Agentic Workflows** | `Workflow` | Implementing multi-turn dialogs, ReAct workflows, or domain-specific agent training capabilities (e.g., Coding, Math). |
| **RL Algorithms** | `Algorithm` | Integrating new RL algorithms (e.g., RLOO, GSPO) or optimizing loss functions and advantage estimations. |
| **Data & Experience** | `Operator`, `Selector` | Designing data cleaning, selection, reward modeling, or experience augmentation and replay strategies. |
| **Use Cases** | `Examples` | Sharing new usages and improving built-in demonstrated configurations. |
| **General Utility** | `docs/`, `tests/` | Improving documentation, adding translations, fixing bugs, or enhancing CLI/GUI tools. |

## How to Start: The "Plugin-First" Approach

To minimize friction and keep the core codebase stable, we recommend the **Plugin-First** workflow for new features:

1. **Develop**: Create your custom module in the `trinity/plugins/` directory.
2. **Auto-Load**: Trinity-RFT automatically detects and registers modules in this directory at runtime without requiring changes to the framework's internal code.
3. **Integrate**: Once your feature is stable and verified, submit a Pull Request to "graduate" your code from `plugins/` into the formal modules (e.g., `trinity/algorithm/` or `trinity/common/workflows/`).

## Submission Checklist

To ensure a smooth review process, please complete the following:

1. **Registration**: If moving code from a plugin to the core framework, register it in the corresponding `__init__.py` mapping.
2. **Testing**: Add or update unit tests in the `tests/` directory. Verify your changes by running:
   ```bash
   python -m pytest tests/
   ```
3. **Code Style**: We use `pre-commit` to maintain code quality. Run the following before committing:
   ```bash
   pre-commit run --all-files
   ```
4. **Description**: Provide a clear PR title and a description that explains the **motivation** (why this change is needed) and the **implementation** (how it works).

---

## Additional Guidelines

- **Bug Reports & Feature Requests**: Please use [GitHub Issues](https://github.com/modelscope/Trinity-RFT/issues). For bugs, include reproduction steps, environment info, and error logs.
- **Major Changes**: For significant architectural changes or large features, please open an issue first to discuss the design with the maintainers.
- **Documentation**: We highly value improvements to our tutorials, docstrings, and translations.

*For a deep dive into the framework's architecture, please refer to the [Full Doc](https://modelscope.github.io/Trinity-RFT/en/main/index.html).*

**Thank you for helping us build a better Reinforcement Fine-Tuning framework!**
