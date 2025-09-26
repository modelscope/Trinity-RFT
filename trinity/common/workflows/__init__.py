# -*- coding: utf-8 -*-
"""Workflow module"""
from trinity.common.workflows.agentscope.react.react_workflow import (
    AgentScopeReActWorkflow,
)
from trinity.common.workflows.customized_math_workflows import (
    AsyncMathBoxedWorkflow,
    MathBoxedWorkflow,
)
from trinity.common.workflows.customized_toolcall_workflows import ToolCallWorkflow
from trinity.common.workflows.envs.agentscope.agentscopev0_react_workflow import (  # will be deprecated soon
    AgentScopeV0ReactMathWorkflow,
)
from trinity.common.workflows.envs.agentscope.agentscopev1_react_workflow import (
    AgentScopeReactMathWorkflow,
)
from trinity.common.workflows.envs.agentscope.agentscopev1_search_workflow import (
    AgentScopeV1ReactSearchWorkflow,
)
from trinity.common.workflows.envs.alfworld.alfworld_workflow import (
    AlfworldWorkflow,
    StepWiseAlfworldWorkflow,
)
from trinity.common.workflows.envs.alfworld.RAFT_alfworld_workflow import (
    RAFTAlfworldWorkflow,
)
from trinity.common.workflows.envs.alfworld.RAFT_reflect_alfworld_workflow import (
    RAFTReflectAlfworldWorkflow,
)
from trinity.common.workflows.envs.email_searcher.workflow import EmailSearchWorkflow
from trinity.common.workflows.envs.sciworld.sciworld_workflow import SciWorldWorkflow
from trinity.common.workflows.envs.webshop.webshop_workflow import WebShopWorkflow
from trinity.common.workflows.eval_workflow import (
    AsyncMathEvalWorkflow,
    MathEvalWorkflow,
)
from trinity.common.workflows.math_rm_workflow import (
    AsyncMathRMWorkflow,
    MathRMWorkflow,
)
from trinity.common.workflows.math_ruler_workflow import (
    AsyncMathRULERWorkflow,
    MathRULERWorkflow,
)
from trinity.common.workflows.math_trainable_ruler_workflow import (
    MathTrainableRULERWorkflow,
)
from trinity.common.workflows.simple_mm_workflow import (
    AsyncSimpleMMWorkflow,
    SimpleMMWorkflow,
)
from trinity.common.workflows.workflow import (
    WORKFLOWS,
    AsyncMathWorkflow,
    AsyncSimpleWorkflow,
    MathWorkflow,
    SimpleWorkflow,
    Task,
    Workflow,
)

__all__ = [
    "Task",
    "Workflow",
    "WORKFLOWS",
    "AsyncSimpleWorkflow",
    "SimpleWorkflow",
    "AsyncMathWorkflow",
    "MathWorkflow",
    "WebShopWorkflow",
    "AlfworldWorkflow",
    "StepWiseAlfworldWorkflow",
    "RAFTAlfworldWorkflow",
    "RAFTReflectAlfworldWorkflow",
    "SciWorldWorkflow",
    "AsyncMathBoxedWorkflow",
    "MathBoxedWorkflow",
    "AsyncMathRMWorkflow",
    "MathRMWorkflow",
    "ToolCallWorkflow",
    "AsyncMathEvalWorkflow",
    "MathEvalWorkflow",
    "AgentScopeV0ReactMathWorkflow",  # will be deprecated soon
    "AgentScopeReactMathWorkflow",
    "AgentScopeV1ReactSearchWorkflow",
    "AgentScopeReActWorkflow",
    "EmailSearchWorkflow",
    "AsyncMathRULERWorkflow",
    "MathRULERWorkflow",
    "MathTrainableRULERWorkflow",
    "AsyncSimpleMMWorkflow",
    "SimpleMMWorkflow",
]
