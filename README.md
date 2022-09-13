# discrete_diffusion

<p align="center">
    <a href="https://github.com/IreneTallini/discrete_diffusion/actions/workflows/test_suite.yml"><img alt="CI" src=https://img.shields.io/github/workflow/status/IreneTallini/discrete_diffusion/Test%20Suite/main?label=main%20checks></a>
    <a href="https://IreneTallini.github.io/discrete_diffusion"><img alt="Docs" src=https://img.shields.io/github/deployments/IreneTallini/discrete_diffusion/github-pages?label=docs></a>
    <a href="https://github.com/grok-ai/nn-template"><img alt="NN Template" src="https://shields.io/badge/nn--template-0.2.1-emerald?style=flat&labelColor=gray"></a>
    <a href="https://www.python.org/downloads/"><img alt="Python" src="https://img.shields.io/badge/python-3.9-blue.svg"></a>
    <a href="https://black.readthedocs.io/en/stable/"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
</p>

torch implementation of discrete diffusion generative models as in https://arxiv.org/abs/2107.03006


## Installation

```bash
pip install git+ssh://git@github.com/IreneTallini/discrete_diffusion.git
```


## Quickstart

[comment]: <> (> Fill me!)


## Development installation

Setup the development environment: tested with python 3.9, torch 1.12, cuda 11.6

```bash
git clone git@github.com:IreneTallini/discrete_diffusion.git
cd discrete_diffusion
conda env create -f env.yaml
conda activate discrete_diffusion
pre-commit install
```
Install torch geometric by hand: https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html

Run the tests:

```bash
pre-commit run --all-files
pytest -v
```


### Update the dependencies

Re-install the project in edit mode:

```bash
pip install -e .[dev]
```
