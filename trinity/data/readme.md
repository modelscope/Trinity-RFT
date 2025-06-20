# Data Engine for RFT Framework

A data processing engine designed for Reinforcement Fine-Tuning (RFT) of Large Language Models.

## Key Features

- Multiple data input types support
  - Meta-prompts and demo examples
  - Raw text corpus
  - Raw QA conversations
  - Raw data with reward functions
  - Tagged data with rewards

- Comprehensive data processing abilities
  - Data cleaning and filtering
  - Data synthesis and augmentation
  - Reward tagging (e.g., in off-line RL) and formatting
  - Active learning iteration

- Integration with [Data-Juicer](https://github.com/modelscope/data-juicer)
  - Reuse Data-Juicer's fruitful operators and high-performance engines
  - Compatible with DJ's configuration system
  - Extend DJ with RFT-specific features, but no need to know DJ's details

- RFT-specific enhancements
  - Task specification parsing
  - Data lineage tracking
  - Feedback loop integration
  - Small model feedback agents

## Usage

### Data Loading and Conversion to `TaskSet`

```python
from trinity.common.rewards import AccuracyReward
from trinity.common.workflows import MathWorkflow
from trinity.common.config import DataProcessorConfig
from trinity.data.core.dataset import RftDataset
from trinity.data.core.formatter import BoxedMathAnswerFormatter, RLHFFormatter

data_config: DataProcessorConfig = ...

# initialize the dataset according to the data config
dataset = RftDataset(data_config)

# format it for the target data and training format
# e.g. format for a boxed-tagged MATH data and RLHF format
dataset.format([
  BoxedMathAnswerFormatter(data_config.format),
  RLHFFormatter(data_config.format),
])

# convert to a task set with global reward function and workflow
task_set = dataset.to_taskset(
  reward_fn=AccuracyReward,
  workflow=MathWorkflow,
)

# downstream usages: convert to experience, training, ...
...

```

### Data Processing

```python
from rft.data_engine import DataTaskParser, DataCleaner, DataSynthesizer

# Parse task config
parser = DataTaskParser(rft_config)
dj_config = parser.parse_to_dj_config()[0]

# Clean data
cleaner = DataCleaner(dj_config)
clean_data = cleaner.process(raw_data)

# Synthesize more data
synthesizer = DataSynthesizer(dj_config)
synth_data = synthesizer.process(clean_data)
```

### Service-based Calling

- You can either run `scripts/start_servers.py` or run `trinity/data/server.py` to start the data server.
  - Before running this config file, you need to replace the `username` and `db_name` with your own username and database name.
  - When requesting it, the server will load the dataset, clean it, compute priority scores from different dimensions, and export the result dataset to the database.
- Then you need to prepare the `data_processor` section in the config file (e.g. [test_cfg.yaml](tests/test_configs/active_iterator_test_cfg.yaml))
  - For the `dj_config_path` argument in it, you can either specify a data-juicer config file path (e.g. [test_dj_cfg.yaml](tests/test_configs/active_iterator_test_dj_cfg.yaml)), or write the demand in `dj_process_desc` argument in natural language and our agent will help you to organize the data-juicer config.
- Finally you can send requests to the data server to start an active iterator to process datasets in many ways:
  - Request with `curl`: `curl "http://127.0.0.1:5005/data_processor/task_pipeline?configPath=tests%2Ftest_configs%2Factive_iterator_test_cfg.yaml"`
  - Request using our simple client:

  ```python
  from trinity.cli.client import request

  res = request(
    url="http://127.0.0.1:5005/data_processor/task_pipeline",
    configPath="tests/test_configs/active_iterator_test_cfg.yaml"
  )

  # downstream processing with res
  ...
  ```

  - Or other implementations.

## Code Structure (related parts)
```text
data/
├── controllers/      # High-level control flows
├── core/             # Core data structures and utils
└── processors/       # Data processing operators
```
