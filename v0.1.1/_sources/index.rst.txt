.. Trinity-RFT documentation master file, created by
   sphinx-quickstart on Thu Apr 17 15:22:13 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Trinity-RFT's documentation!
=======================================

.. include:: main.md
   :parser: myst_parser.sphinx_


.. toctree::
   :maxdepth: 1
   :glob:
   :hidden:
   :caption: Tutorial

   tutorial/example_reasoning_basic.md
   tutorial/example_reasoning_advanced.md
   tutorial/example_async_mode.md
   tutorial/example_multi_turn.md
   tutorial/example_dpo.md
   tutorial/example_data_functionalities.md
   tutorial/trinity_configs.md
   tutorial/trinity_programming_guide.md

.. toctree::
   :maxdepth: 1
   :glob:
   :caption: API Reference

   build_api/trinity.buffer
   build_api/trinity.explorer
   build_api/trinity.trainer
   build_api/trinity.manager
   build_api/trinity.common
   build_api/trinity.utils
