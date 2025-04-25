# -*- coding: utf-8 -*-
"""Workflow module"""
from .envs.alfworld.alfworld_workflow import AlfworldWorkflow
from .envs.sciworld.sciworld_workflow import SciWorldWorkflow
from .envs.webshop.webshop_workflow import WebShopWorkflow
from .workflow import WORKFLOWS, MathWorkflow, SimpleWorkflow

__all__ = [
    "WORKFLOWS",
    "SimpleWorkflow",
    "MathWorkflow",
    "WebShopWorkflow",
    "AlfworldWorkflow",
    "SciWorldWorkflow",
]
