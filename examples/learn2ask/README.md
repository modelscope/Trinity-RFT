# Learn2Ask: Getting Started

This guide demonstrates how to train a proactive LLM using the **Learn2Ask** framework from [Grounded in Reality: Learning and Deploying Proactive LLM from Offline Logs](https://arxiv.org/abs/2510.25441).
**Hardware requirement**: ≥32 H20 (or equivalent) GPUs for full-scale reproduction.

All relevant files are located under `examples/learn2ask/`:
- Workflow & prompts: `examples/learn2ask/workflow/`
- Training config: `examples/learn2ask/train.yaml`
- Data preparation scripts: `examples/learn2ask/data_prepare/`

---

## Step 1. Prepare Datasets

Download the [RealMedConv](https://huggingface.co/datasets/datajuicer/RealMedConv) dataset (`.jsonl` format). Each line is a full conversation log:

```json
{
  "session_id": 35310,
  "diagn": "Upper Respiratory Tract Infection",
  "messages": [{"role": "user", "content": "Sore throat, phlegm, red eyes, cough, hoarse voice"}, {"role": "user", "content": "I took Amoxicillin"}, {"role": "user", "content": "But I still don't feel well"}, {"role": "user", "content": "Mainly it's a respiratory infection, sore throat, phlegm, hoarse voice, red eyes"}, {"role": "user", "content": "When I wake up, there is a lot of eye discharge, and a lot of phlegm"}, {"role": "assistant", "content": "How long have the symptoms been present?"}, {"role": "user", "content": "About 2 days"}, {"role": "user", "content": "My eyes are very red"}, {"role": "assistant", "content": "Is there any discharge?"}, {"role": "user", "content": "Yes"}, {"role": "user", "content": "Please check my description, I wrote all the details"}, {"role": "assistant", "content": "Sure"}, {"role": "assistant", "content": "The internet was down just now"}, {"role": "user", "content": "Okay"}, {"role": "assistant", "content": "Is the discharge thick, thin, or stringy?"}, {"role": "user", "content": "It's thick"}, {"role": "user", "content": "Yellowish"}, {"role": "user", "content": "Mainly a lot in the morning, and a lot of phlegm"}, {"role": "assistant", "content": "Does it affect your vision? Do you have eye pain? Itchy eyes? Foreign body sensation? Tears?"}, {"role": "user", "content": "No"}, {"role": "user", "content": "Mainly still sore throat"}, {"role": "user", "content": "The eyes are just red and have discharge"}, {"role": "user", "content": "Sore throat, a lot of phlegm, mild cough, hoarse voice"}, {"role": "assistant", "content": "Okay"}, {"role": "assistant", "content": "Have you had any medical examinations or medication history? Any history of drug allergies or chronic diseases?"}, {"role": "user", "content": "No"}, {"role": "user", "content": "Please help as soon as possible, it's getting late"}, {"role": "assistant", "content": "<med_search>"}]
}
```
You need to perform the following preprocessing steps to turn the log in to training/testing samples for our learn2ask framework, there are two simple steps:
- Segment the original conversation log (session) into context–future pairs, then extract `info_truth` labels from the `remaining_chat` field.
```bash
python examples/learn2ask/workflow/data_prepare/1_info_extract_pipeline.py
```

- Convert these samples into final training/testing datasets.
```bash
examples/learn2ask/workflow/data_prepare/2_build_dataset.py
```

These scripts are implementations of the following procedures.

### Segment into Context–Future Pairs
For each turn in a session, split the conversation into:
- `messages`: the **observed context** up to that point,
- `remaining_chat`: the **subsequent dialogue** (i.e., the "future" from that turn onward).
Each segmented sample should include a unique `cid` (e.g., `{session_id}_{turn_index}`).
```JSON
{
   "cid": "35310_7",
   "session_id": "35310",
   "diagn": "Upper Respiratory Tract Infection",
   "messages": [{"role": "user", "content": "Sore throat, phlegm, red eyes, cough, hoarse voice"}, {"role": "user", "content": "I took Amoxicillin"}, {"role": "user", "content": "But I still don't feel well"}, {"role": "user", "content": "Mainly it's a respiratory infection, sore throat, phlegm, hoarse voice, red eyes"}, {"role": "user", "content": "When I wake up, there is a lot of eye discharge, and a lot of phlegm"}, {"role": "assistant", "content": "How long have the symptoms been present?"}, {"role": "user", "content": "About 2 days"}, {"role": "user", "content": "My eyes are very red"}],
   "remaining_chat": [{"role": "assistant", "content": "Is there any discharge?"}, {"role": "user", "content": "Yes"}, {"role": "user", "content": "Please check my description, I wrote all the details"}, {"role": "assistant", "content": "Sure"}, {"role": "assistant", "content": "The internet was down just now"}, {"role": "user", "content": "Okay"}, {"role": "assistant", "content": "Is the discharge thick, thin, or stringy?"}, {"role": "user", "content": "It's thick"}, {"role": "user", "content": "Yellowish"}, {"role": "user", "content": "Mainly a lot in the morning, and a lot of phlegm"}, {"role": "assistant", "content": "Does it affect your vision? Do you have eye pain? Itchy eyes? Foreign body sensation? Tears?"}, {"role": "user", "content": "No"}, {"role": "user", "content": "Mainly still sore throat"}, {"role": "user", "content": "The eyes are just red and have discharge"}, {"role": "user", "content": "Sore throat, a lot of phlegm, mild cough, hoarse voice"}, {"role": "assistant", "content": "Okay"}, {"role": "assistant", "content": "Have you had any medical examinations or medication history? Any history of drug allergies or chronic diseases?"}, {"role": "user", "content": "No"}, {"role": "user", "content": "Please help as soon as possible, it's getting late"}, {"role": "assistant", "content": "<med_search>"}]
}
```
### Extract ground-truth labels for rewards
From `remaining_chat`, derive the following new fields:
- `decision_truth`: the correct action (e.g., `"continue"` or `"stop"`),
- `info_truth`: structured symptom information used for reward computation
These ground truth are used to evaluate the rewards in training, e.g., $R_a$ and $R_s$.
```JSON
{
   "cid": "35310_7",
   "session_id": "35310",
   "diagn": "Upper Respiratory Tract Infection",
   "messages": [{"role": "user", "content": "Sore throat, phlegm, red eyes, cough, hoarse voice"}, {"role": "user", "content": "I took Amoxicillin"}, {"role": "user", "content": "But I still don't feel well"}, {"role": "user", "content": "Mainly it's a respiratory infection, sore throat, phlegm, hoarse voice, red eyes"}, {"role": "user", "content": "When I wake up, there is a lot of eye discharge, and a lot of phlegm"}, {"role": "assistant", "content": "How long have the symptoms been present?"}, {"role": "user", "content": "About 2 days"}, {"role": "user", "content": "My eyes are very red"}],
   "remaining_chat": [{"role": "assistant", "content": "Is there any discharge?"}, {"role": "user", "content": "Yes"}, {"role": "user", "content": "Please check my description, I wrote all the details"}, {"role": "assistant", "content": "Sure"}, {"role": "assistant", "content": "The internet was down just now"}, {"role": "user", "content": "Okay"}, {"role": "assistant", "content": "Is the discharge thick, thin, or stringy?"}, {"role": "user", "content": "It's thick"}, {"role": "user", "content": "Yellowish"}, {"role": "user", "content": "Mainly a lot in the morning, and a lot of phlegm"}, {"role": "assistant", "content": "Does it affect your vision? Do you have eye pain? Itchy eyes? Foreign body sensation? Tears?"}, {"role": "user", "content": "No"}, {"role": "user", "content": "Mainly still sore throat"}, {"role": "user", "content": "The eyes are just red and have discharge"}, {"role": "user", "content": "Sore throat, a lot of phlegm, mild cough, hoarse voice"}, {"role": "assistant", "content": "Okay"}, {"role": "assistant", "content": "Have you had any medical examinations or medication history? Any history of drug allergies or chronic diseases?"}, {"role": "user", "content": "No"}, {"role": "user", "content": "Please help as soon as possible, it's getting late"}, {"role": "assistant", "content": "<med_search>"}],
   "decision_truth": "continue",
   "info_truth": "Symptom: sore throat, Symptom quality: thick discharge, Symptom quality: yellowish discharge, Symptom quality: a lot of phlegm, Symptom severity: mild cough, Symptom quality: hoarse voice"
}
```

---

## Step 2. Configure and Train
Update `examples/learn2ask/train.yaml` with paths to:
- Your processed datasets,
- Base model,
- Checkpoint output directory.

Then, launch training:
```bash
trinity run --config examples/learn2ask/train.yaml --plugin-dir examples/learn2ask/workflow
````
---

## Step 3. Evaluate
Use the rollout-n-evaluate pipeline:
- Generate responses on the test set (e.g. with *vLLM*),
- Evaluate outputs using **`qwen2.5-32b-instruct`** with the evaluator.

You may configure the settings then run the pipeline by launching:
```bash
python examples/learn2ask/workflow/data_prepare/3_rollout_then_evaluate.py
```
