# GRPO with VLM

This example shows the usage of GRPO with Qwen2.5-VL-3B-Instruct on the [geometry3k](https://huggingface.co/datasets/hiyouga/geometry3k) dataset.
The specific requirements are:

```yaml
vllm>=0.9.1,<0.10.0
transformers<4.53.0
qwen_vl_utils
```

For other detailed information, please refer to the [documentation](../../docs/sphinx_doc/source/tutorial/example_reasoning_basic.md).

The config files are located in [`vlm.yaml`](vlm.yaml) and [`train_vlm.yaml`](train_vlm.yaml).
