# Trinity-RFT Documentation

Please use the following commands to build sphinx doc of Trinity-RFT.

```shell
# Step 1: install dependencies

# for bash
pip install -e .[doc]
# for zsh
pip install -e .\[doc\]

# Step 2: build sphinx doc

cd docs/sphinx_doc
# build docs for all existing tags and main branch
./build_doc.sh
# build docs for the current branch
./build_doc.sh --branch <current_branch_name>
```

The sphinx doc is built in `docs/sphinx_doc/build/html`.
