# -*- coding: utf-8 -*-
"""Workflow module"""
from .envs.alfworld.alfworld_workflow import AlfworldWorkflow
from .envs.sciworld.sciworld_workflow import SciWorldWorkflow
from .envs.webshop.webshop_workflow import WebShopWorkflow
from .workflow import WORKFLOWS, MathWorkflow, SimpleWorkflow, Task

__all__ = [
    "Task",
    "WORKFLOWS",
    "SimpleWorkflow",
    "MathWorkflow",
    "WebShopWorkflow",
    "AlfworldWorkflow",
    "SciWorldWorkflow",
]
