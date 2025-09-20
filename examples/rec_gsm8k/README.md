# Example: REC on GSM8k dataset

This example shows the usage of REC on the [GSM8k dataset](https://huggingface.co/datasets/openai/gsm8k).

For more detailed information, please refer to the [documentation](../../docs/sphinx_doc/source/tutorial/example_reasoning_basic.md).

The config files are located in [`gsm8k.yaml`](gsm8k.yaml) and [`train_gsm8k.yaml`](train_gsm8k.yaml).

# Group-relative REINFORCE Families
This folder provides **example configurations** for running different group-relative REINFORCE families within Trinity-RFT.

It includes three major families:

- **REC family** (clipping + importance sampling)
- **REP family** (regularization-based variants)
- **RED family** (data-distribution shaping strategies)

We also provide baseline implementations such as **Vanilla REINFORCE** and **GRPO**.

All algorithms are instantiated through modular YAML configs for easy reproduction and extension.

# Summary Table 📝

| Family        | Variants                                        | Key Idea                            |
| ------------- | ----------------------------------------------- | ----------------------------------- |
| **Baselines** | REINFORCE, GRPO                                 | Standard references                 |
| **REC**       | OneSide-NoIS, OneSide-IS, TwoSide-IS, Ring-NoIS | Clipping + importance sampling      |
| **REP**       | AsymRE, OPMD                                    | Regularization |
| **RED**       | Drop, Weight                                    | Data-distribution shaping           |



# Instantiations

## Baselines

### REINFORCE
Vanilla REINFORCE with group mean as baseline.

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

### GRPO
GRPO implemented with zero KL regularizer. Regularization can be enabled via `kl_loss_fn` and `kl_loss_fn_args`.

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

## REC family
Variants of clipping and importance-sampling strategies.
- REC-OneSide-NoIS
- REC-OneSide-IS
- REC-TwoSide-IS
- REC-Ring-NoIS

### REC-OneSide-NoIS

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

### REC-OneSide-IS

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

### REC-TwoSide-IS

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
### REC-Ring-NoIS

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

## REP family

Regularization-based algorithms.
- AsymRE (forward KL regularization)
- Kimi’s OPMD (k2 regularizer)

### AsymRE

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


### Kimi's OPMD

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

## RED family
Data-distribution shaping variants.
- RED-Drop (drop extra negative examples to balance the positive examples v.s. negative examples)
- RED-Weight (advantage-weighting strategy)

### RED-Drop

```
algorithm:
  algorithm_type: rec
  policy_loss_fn_args:
    clip_mode: "none"
    weight: "none"
    regularizer: "none"
  advantage_fn_args:
    std_normalize: false
    drop: balance
```


### RED-Weight

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
