# Factual Recall Dynamics

This repository contains code for investigating factual recall dynamics in language models, particularly focusing on how models retain and recall factual information during training and fine-tuning.

## Features

- Knowledge graph generation with configurable attributes
- Synthetic data generation for pre-training and fine-tuning
- Name tokenization experiments (whole vs. subword)
- Fact recall evaluation framework
- SLURM cluster support for large-scale experiments

## Installation

```bash
git clone https://github.com/manncodes/factual-recall-dynamics.git
cd factual-recall-dynamics
pip install -r requirements.txt
```

## Usage

### Configuration

Edit the configuration files in the `config/` directory to customize your experiments:

```bash
cp config/default.yaml config/my_experiment.yaml
# Edit my_experiment.yaml with your desired settings
```

### Running Experiments

To run a basic experiment:

```bash
python scripts/run_experiment.py --config config/my_experiment.yaml
```

For SLURM clusters:

```bash
sbatch scripts/run_slurm.sh config/slurm.yaml
```

### Data Generation

Generate synthetic knowledge graph and training data:

```bash
python scripts/prepare_data.py --num_entities 1000 --name_mode subword_prefix
```

## Experiment Design

This codebase allows you to explore several research questions:

1. How do models learn to associate factual information with entities?
2. What is the effect of tokenization strategy on factual recall?
3. How does factual information compete or interfere during training?
4. What is the dynamics of factual recall during pre-training vs. fine-tuning?

## Directory Structure

- `config/`: Configuration files
- `data/`: Generated data and resources
- `scripts/`: Executable scripts for various tasks
- `src/`: Core source code
  - `data/`: Data generation and processing modules
  - `models/`: Model training components
  - `evaluation/`: Evaluation metrics and test generation
  - `utils/`: Utility functions
- `tests/`: Unit tests

## License

[MIT License](LICENSE)

## Citation

If you use this code in your research, please cite:

```
@misc{factual-recall-dynamics,
  author = {Your Name},
  title = {Factual Recall Dynamics},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/manncodes/factual-recall-dynamics}
}
```
