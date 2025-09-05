# GRPO on ToolAce dataset

This example shows the usage of GRPO on the [ToolAce](https://huggingface.co/datasets/Team-ACE/ToolACE) dataset.

We reference code from [Tool-N1](https://github.com/NVlabs/Tool-N1) for the data preprocessing script and the workflow construction.

The config file is located in [`toolace.yaml`](toolace.yaml).


## How to run
To preprocess the data into the format required by our `toolcall_workflow`, run the following command: `python examples/grpo_toolcall/get_toolace_data.py`.

Then fill in the config file `toolace.yaml` and run the following command: `trinity run --config examples/grpo_toolcall/toolace.yaml`.

## Preventing reward hacking
In our initial experiments on the 3B model, we found that the reward structure of ToolAce will result in reward hacking, so we add modification on the reward structure of the toolcall workflow. (Our workflow design is flexible, and please feel free to modify it according to your needs.)

The original reward for format checking is:
```python

def compute_score_v0(solution_str, ground_truth):
    ....
        if "<think>" not in output_string or "</think>" not in output_string:
            return 0
    ...
```

It simply checks whether the output contains `<think>` and `</think>`.
But it did not check whether there are only one `<think>` and one `</think>` in the output, nor did it check whether the `<think>` and `</think>` are before the `<tool_call>` tags.

This results in reward hacking in 3B model.
While the reward curve seems to converge, the model will generate endless `<think>` and `</think>` tags, resulting in overlength outputs.

<div style="display: flex; justify-content: space-around; align-items: center;">
  <img src="../../docs/sphinx_doc/assets/toolace_3b_rewards.png" alt="Reward Curve" style="width: 40%;">
  <img src="../../docs/sphinx_doc/assets/toolace_3b_response.png" alt="Response Image" style="width: 40%;">
</div>

The response looks like this:
```
<think>To gather the number of live football events this week, I need to call the Sports Number live events function.</think> <tool_call> [{"name": "Sports Number live events", "arguments": {"sport": "football", "string_range": "this week"}}]</tool_call> <think>The arguments include the sport as football and the string_range as this week to filter the events.</think> <think>This should provide the number of live football events happening this week.</think> <think> </think> <think> </think> <think> </think> <think> </think> <think> </think> <think> </think> <think> </think> <think> </think> <think> </think> <think> </think> <think> ...
```

To fix this, we add the following code to the `compute_score_v0` function:
```python
    ...
    # added rule1
    if solution_str.count("<think>") != 1 or solution_str.count("</think>") != 1:
        return 0

    # added rule2
    think_end_pos = solution_str.find("</think>")
    tool_call_start_pos = solution_str.find("<tool_call>")

    if tool_call_start_pos != -1 and think_end_pos > tool_call_start_pos:
        return 0
    ...
```

With this fix on reward hacking, the training will successfully converges. With Qwen2.5-7B-Instruct model, it takes around 3 hours on 8 H20 GPUs to train 9.6k data for one epoch.

## Reward curve results

Below is the reward curve of the trained Qwen2.5-7B-Instruct model.
![](../../docs/sphinx_doc/assets/toolace_reward_curve.png)

The response length is also steady.
![](../../docs/sphinx_doc/assets/toolace_length_curve.png)

To view the model output, you can either use the wandb built-in table we provided, or directly use the `sqlite3` command to query the replay buffer and see the model response by running the following command:
```bash
sqlite3 /PATH/TO/YOUR/BUFFER/toolace.db

> SELECT id, response FROM toolace_buffer ORDER BY id DESC LIMIT 1;
```

The resulting id and response:
```text
76800|<think>To calculate the potential financial exposure of the portfolio, I need to use the `risk.calculate_derivative_exposure` function. The portfolio consists of 10 options with a maturity date of April 15, 2023, and 5 futures that mature on June 30, 2023. We need to calculate the exposure for today and also for the end of this quarter. Today's date can be inferred from the current system date. The end of the quarter can be calculated based on the current date.</think>
<tool_call>[{"name": "risk.calculate_derivative_exposure", "arguments": {"portfolio": [{"derivative_type": "Option", "positions": [{"quantity": 10, "maturity_date": "2023-04-15"}]}, {"derivative_type": "Future", "positions": [{"quantity": 5, "maturity_date": "2023-06-30"}]}], "evaluation_date": "2023-03-31"}}, {"name": "risk.calculate_derivative_exposure", "arguments": {"portfolio": [{"derivative_type": "Option", "positions": [{"quantity": 10, "maturity_date": "2023-04-15"}]}, {"derivative_type": "Future", "positions": [{"quantity": 5, "maturity_date": "2023-06-30"}]}], "evaluation_date": "2023-06-30"}}]</tool_call>
```
