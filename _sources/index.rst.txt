.. Trinity-RFT documentation master file, created by
   sphinx-quickstart on Thu Apr 17 15:22:13 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Trinity-RFT's documentation!
=======================================

.. include:: tutorial/main.md
   :parser: myst_parser.sphinx_


.. toctree::
   :maxdepth: 1
   :glob:
   :hidden:
   :caption: Tutorial

   tutorial/example_reasoning_basic.md
   tutorial/example_reasoning_advanced.md
   tutorial/example_multi_turn.md
   tutorial/example_dpo.md
   tutorial/example_data_functionalities.md
   tutorial/trinity_configs.md
   tutorial/trinity_programming_guide.md

.. toctree::
   :maxdepth: 2
   :glob:
   :caption: API Reference

   example.md

   trinity.buffer.reader
   trinity.buffer
   trinity.buffer.schema
   trinity.buffer.writer
   trinity.common.models
   trinity.common.rewards
   trinity.common
   trinity.common.workflows
   trinity.explorer
   trinity.manager
   trinity
   trinity.trainer
   trinity.trainer.verl
   trinity.utils


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
