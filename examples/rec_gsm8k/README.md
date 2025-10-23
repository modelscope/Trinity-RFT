# Example: group-relative REINFORCE variants on GSM8k dataset

This example shows the usage of group-relative REINFORCE variants on the [GSM8k dataset](https://huggingface.co/datasets/openai/gsm8k).

For more details about algorithm design, please refer to [our paper](https://arxiv.org/abs/2509.24203).

The config file is located in [`gsm8k.yaml`](gsm8k.yaml).

## Group-relative REINFORCE variants

This folder provides example configurations for running different group-relative REINFORCE variants within Trinity-RFT.
It includes three major families:

- **REC family** (regularization by clipping)
- **REP family** (regularization by an additive loss term)
- **RED family** (actively shaping data distribution)

These include baseline algorithms like vanilla REINFORCE and GRPO as special cases.

All algorithms are instantiated through modular YAML configs for easy reproduction and extension.

## Summary Table 📝

| Family        | Variants                                        | Key Idea                            |
| ------------- | ----------------------------------------------- | ----------------------------------- |
| **Baselines** | REINFORCE, GRPO                                 | Standard references          |
| **REC**       | OneSide/TwoSide/Ring-IS/NoIS                    | Clipping as regularization, with or without importance sampling   |
| **REP**       | AsymRE, OPMD                                    | Regularization by an additive loss term |
| **RED**       | Drop, Weight                                    | Actively shaping data distribution      |



## Instantiations

### Baselines

**Vanilla REINFORCE** with group mean as baseline:

```
algorithm:
  algorithm_type: rec
  policy_loss_fn_args:
    epsilon_low: 0.2
    epsilon_high: 0.2
    clip_mode: "none" # no clipping
    weight: "none" # uniform weighting for samples
    temp: 1.0
    regularizer: "none" # no regularizer
    regularizer_coef: 0.0
  advantage_fn_args:
    std_normalize: false
```

**GRPO** with KL regularization (enabled via `kl_loss_fn` and `kl_loss_fn_args`):

```
algorithm:
  algorithm_type: rec
  policy_loss_fn_args:
    epsilon_low: 0.2
    epsilon_high: 0.2
    clip_mode: "one-side"
    weight: "importance_sampling"
    temp: 1.0
    regularizer: "none"
    regularizer_coef: 0.0
  advantage_fn_args:
    std_normalize: true
  kl_loss_fn: 'k2'
  kl_loss_fn_args:
    kl_coef:  0.0
```

### REC family

**REC-OneSide-NoIS:**

```
algorithm:
  algorithm_type: rec
  policy_loss_fn_args:
    epsilon_low: 0.2
    epsilon_high: 0.2
    clip_mode: "one-side"
    weight: "none"
    temp: 1.0
    regularizer: "none"
    regularizer_coef: 0.0
  advantage_fn_args:
    std_normalize: false
```

**REC-OneSide-IS:**

```
algorithm:
  algorithm_type: rec
  policy_loss_fn_args:
    epsilon_low: 0.2
    epsilon_high: 0.2
    clip_mode: "one-side"
    weight: "importance_sampling"
    temp: 1.0
    regularizer: "none"
    regularizer_coef: 0.0
  advantage_fn_args:
    std_normalize: false
```

**REC-TwoSide-IS:**

```
algorithm:
  algorithm_type: rec
  policy_loss_fn_args:
    epsilon_low: 0.2
    epsilon_high: 0.2
    clip_mode: "two-side"
    weight: "importance_sampling"
    temp: 1.0
    regularizer: "none"
    regularizer_coef: 0.0
  advantage_fn_args:
    std_normalize: false
```

**REC-Ring-NoIS:**

```
algorithm:
  algorithm_type: rec
  policy_loss_fn_args:
    epsilon_low: 0.2
    epsilon_high: 0.2
    epsilon_low_prime: 0.6
    epsilon_high_prime: 2.0
    clip_mode: "ring"
    weight: "none"
    temp: 1.0
    regularizer: "none"
    regularizer_coef: 0.0
  advantage_fn_args:
    std_normalize: false
```

### REP family


**Meta's AsymRE:**

```
algorithm:
  algorithm_type: rec
  policy_loss_fn_args:
    clip_mode: "none"
    weight: "none"
    temp: 1.0
    regularizer: "forward-kl"
    regularizer_coef: 0.1
  advantage_fn_args:
    std_normalize: false
```


**Kimi's OPMD:**

```
algorithm:
  algorithm_type: rec
  policy_loss_fn_args:
    clip_mode: "none"
    weight: "none"
    regularizer: "k2"
    regularizer_coef: 0.1
  advantage_fn_args:
    std_normalize: false
```

### RED family


**RED-Drop:**

```
algorithm:
  algorithm_type: rec
  policy_loss_fn_args:
    clip_mode: "none"
    weight: "none"
    regularizer: "none"
  advantage_fn_args:
    std_normalize: false
    drop: "balance"
```


**RED-Weight:**

```
algorithm:
  algorithm_type: rec
  policy_loss_fn_args:
    clip_mode: "none"
    weight: "advantage"
    regularizer: "none"
    temp: 1.0
  advantage_fn_args:
    std_normalize: false
```

## Citation

```bibtex
@misc{yao2025grouprelativereinforcesecretlyoffpolicy,
      title={Group-Relative REINFORCE Is Secretly an Off-Policy Algorithm: Demystifying Some Myths About GRPO and Its Friends},
      author={Chaorui Yao and Yanxi Chen and Yuchang Sun and Yushuo Chen and Wenhao Zhang and Xuchen Pan and Yaliang Li and Bolin Ding},
      year={2025},
      eprint={2509.24203},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2509.24203},
}
```
