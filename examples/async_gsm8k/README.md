# Asynchronous mode on GSM8K dataset

This example shows the usage of GRPO on the GSM8K dataset in an asynchronous mode.

For more detailed information, please refer to the [documentation](../../docs/sphinx_doc/source/tutorial/example_async_mode.md).

The config files are located in [`trainer.yaml`](trainer.yaml) and [`explorer.yaml`](explorer.yaml).

You can run this example by the following command:

```bash
bash examples/async_gsm8k/run.sh
```

> [!NOTE]
> In the current asynchronous RFT training, it is recommended to start the Trainer before starting the Explorer to avoid the situation where the Trainer cannot read the generated experience data after the Explorer process terminates prematurely. This issue will be resolved in a future version.
