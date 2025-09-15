# -*- coding: utf-8 -*-
"""Workflow module"""
from .customized_math_workflows import MathBoxedWorkflow
from .customized_toolcall_workflows import ToolCallWorkflow
from .envs.agentscope.agentscopev0_react_workflow import (  # will be deprecated soon
    AgentScopeV0ReactMathWorkflow,
)
from .envs.agentscope.agentscopev1_react_workflow import AgentScopeReactMathWorkflow
from .envs.agentscope.agentscopev1_search_workflow import (
    AgentScopeV1ReactSearchWorkflow,
)
from .envs.alfworld.alfworld_workflow import AlfworldWorkflow, StepWiseAlfworldWorkflow
from .envs.alfworld.RAFT_alfworld_workflow import RAFTAlfworldWorkflow
from .envs.alfworld.RAFT_reflect_alfworld_workflow import RAFTReflectAlfworldWorkflow
from .envs.email_searcher.workflow import EmailSearchWorkflow
from .envs.sciworld.sciworld_workflow import SciWorldWorkflow
from .envs.webshop.webshop_workflow import WebShopWorkflow
from .eval_workflow import MathEvalWorkflow
from .math_rm_workflow import MathRMWorkflow
from .math_ruler_workflow import MathRULERWorkflow
from .math_trainable_ruler_workflow import MathTrainableRULERWorkflow
from .simple_mm_workflow import SimpleMMWorkflow
from .workflow import WORKFLOWS, MathWorkflow, SimpleWorkflow, Task, Workflow

__all__ = [
    "Task",
    "Workflow",
    "WORKFLOWS",
    "SimpleWorkflow",
    "MathWorkflow",
    "WebShopWorkflow",
    "AlfworldWorkflow",
    "StepWiseAlfworldWorkflow",
    "RAFTAlfworldWorkflow",
    "RAFTReflectAlfworldWorkflow",
    "SciWorldWorkflow",
    "MathBoxedWorkflow",
    "MathRMWorkflow",
    "ToolCallWorkflow",
    "MathEvalWorkflow",
    "AgentScopeV0ReactMathWorkflow",  # will be deprecated soon
    "AgentScopeReactMathWorkflow",
    "AgentScopeV1ReactSearchWorkflow",
    "EmailSearchWorkflow",
    "MathRULERWorkflow",
    "MathTrainableRULERWorkflow",
    "SimpleMMWorkflow",
]
