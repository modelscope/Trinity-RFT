import torch
import copy

n_groups, skipped_group_num = 5, 10
exp_groups = {i: [i] for i in range(n_groups)}


idx = torch.randint(0, len(exp_groups), (skipped_group_num,)).tolist()
duplicated_groups = [copy.deepcopy(exp_groups[i]) for i in idx]
duplicated_exps = [exp for group in duplicated_groups for exp in group]
exps = [exp for group in exp_groups.values() for exp in group]
print(exps)
print(duplicated_exps)