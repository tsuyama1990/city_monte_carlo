#!bin/bash

pyenv install 3.12
pyenv local 3.12
poetry env use python
poetry install --with dev
poetry run pre-commit install
poetry shell
