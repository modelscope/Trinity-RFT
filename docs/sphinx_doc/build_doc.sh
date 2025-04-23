#!/bin/bash
sphinx-apidoc -f -o source/build_api ../../trinity -t _templates
make clean html
