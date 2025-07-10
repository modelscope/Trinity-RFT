# -*- coding: utf-8 -*-
"""Workflow module"""
from .customized_math_workflows import MathBoxedWorkflow
from .dapo_workflow import MathDAPOWorkflow
from .envs.alfworld.alfworld_workflow import AlfworldWorkflow
from .envs.sciworld.sciworld_workflow import SciWorldWorkflow
from .envs.webshop.webshop_workflow import WebShopWorkflow
from .workflow import WORKFLOWS, MathWorkflow, SimpleWorkflow, Task, Workflow

__all__ = [
    "Task",
    "Workflow",
    "WORKFLOWS",
    "SimpleWorkflow",
    "MathWorkflow",
    "WebShopWorkflow",
    "AlfworldWorkflow",
    "SciWorldWorkflow",
    "MathBoxedWorkflow",
    "MathDAPOWorkflow",
]
