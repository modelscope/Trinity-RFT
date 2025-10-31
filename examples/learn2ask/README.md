# How to Get Started

All experiments were conducted using the [`Trinity-RFT`](https://github.com/modelscope/Trinity-RFT) training framework. You will need at least **32 GPUs** (H20 or more powerful ones). To successfully replicate our results, we recommend following the pipeline.
The prompts and workflow are located in the `examples/learn2ask/plugins/` directory, the training config files is located in `examples/learn2ask/train.yaml`. And the training/testing data with complete processing scripts are located in the `examples/learn2ask/data_prepare/` directory.

---

## Step 1. Data Processing
> 📁 The scripts for this data processing pipeline are located in the `examples/learn2ask/data_prepare/` directory.

The original training data is provided as a `.jsonl` file, where each line represents a complete session.
```JSON
{
   "session_id": 35310,
   "diagn": "Upper Respiratory Tract Infection",
   "messages": [{"role": "user", "content": "Sore throat, phlegm, red eyes, cough, hoarse voice"}, {"role": "user", "content": "I took Amoxicillin"}, {"role": "user", "content": "But I still don't feel well"}, {"role": "user", "content": "Mainly it's a respiratory infection, sore throat, phlegm, hoarse voice, red eyes"}, {"role": "user", "content": "When I wake up, there is a lot of eye discharge, and a lot of phlegm"}, {"role": "assistant", "content": "How long have the symptoms been present?"}, {"role": "user", "content": "About 2 days"}, {"role": "user", "content": "My eyes are very red"}, {"role": "assistant", "content": "Is there any discharge?"}, {"role": "user", "content": "Yes"}, {"role": "user", "content": "Please check my description, I wrote all the details"}, {"role": "assistant", "content": "Sure"}, {"role": "assistant", "content": "The internet was down just now"}, {"role": "user", "content": "Okay"}, {"role": "assistant", "content": "Is the discharge thick, thin, or stringy?"}, {"role": "user", "content": "It's thick"}, {"role": "user", "content": "Yellowish"}, {"role": "user", "content": "Mainly a lot in the morning, and a lot of phlegm"}, {"role": "assistant", "content": "Does it affect your vision? Do you have eye pain? Itchy eyes? Foreign body sensation? Tears?"}, {"role": "user", "content": "No"}, {"role": "user", "content": "Mainly still sore throat"}, {"role": "user", "content": "The eyes are just red and have discharge"}, {"role": "user", "content": "Sore throat, a lot of phlegm, mild cough, hoarse voice"}, {"role": "assistant", "content": "Okay"}, {"role": "assistant", "content": "Have you had any medical examinations or medication history? Any history of drug allergies or chronic diseases?"}, {"role": "user", "content": "No"}, {"role": "user", "content": "Please help as soon as possible, it's getting late"}, {"role": "assistant", "content": "<med_search>"}]
}
```
You need to perform the following preprocessing steps:

### 1. Segment sessions into context–future pairs
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
### 2. Extract ground-truth labels for rewards
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

## Step 2. Training Configuration
> 📁 The training workflow logic and prompt templates are already implemented in `examples/learn2ask/plugins/`.

Using the processed dataset, slightly modify the configuration file:
- Use `train_rft.yaml` for reinforcement fine-tuning (RFT),
- Use `train_sft.yaml` for supervised fine-tuning (SFT) in ablation studies.

You only need to update the paths for:
- Dataset,
- Base model,
- Checkpoint output directory.

---

## Step 3. Evaluation
> 📁 The evaluation pipeline scripts are provided in the `ckpt_evaluation/` folder.

After training, use the saved checkpoints to generate responses on the test set with **vLLM**. Then, evaluate the model outputs using **`qwen2.5-32b-instruct`** as the evaluator.
