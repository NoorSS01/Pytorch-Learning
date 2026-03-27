# PyTorch Learning Phases

This repository is a step-by-step PyTorch learning track built as six Jupyter notebooks. It moves from tensor basics to training neural networks, CNNs, and saving/loading models.

## Repository Structure

- `Phase-1.ipynb`: Tensor fundamentals (creation, shape/dtype, operations, reshape/transpose, randomness and seeds)
- `Phase-2.ipynb`: Autograd and optimization (manual gradient descent and `torch.optim`)
- `Phase-3.ipynb`: Binary classification with a feedforward neural network
- `Phase-4.ipynb`: Regression on a synthetic house-price dataset
- `Phase-5.ipynb`: CNN on synthetic image patterns (vertical vs horizontal)
- `Phase-6.ipynb`: Model persistence (`state_dict` save/load) and prediction consistency check
- `pytorch_model_state.pth`: Saved model weights produced in Phase 6

## Learning Roadmap

1. **Tensor & math basics** in Phase 1
2. **Gradients + optimization loop** in Phase 2
3. **End-to-end classifier training** in Phase 3
4. **Regression workflow + visualization** in Phase 4
5. **Convolutional networks** in Phase 5
6. **Production habit: save/load weights** in Phase 6

## Environment Setup

The notebooks use:

- Python 3.9+ (recommended)
- `torch`
- `numpy`
- `matplotlib`
- `jupyter`

Install quickly:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install torch numpy matplotlib jupyter
```

## How To Run

1. Start Jupyter:

```bash
jupyter notebook
```

2. Open notebooks in order:

- `Phase-1.ipynb`
- `Phase-2.ipynb`
- `Phase-3.ipynb`
- `Phase-4.ipynb`
- `Phase-5.ipynb`
- `Phase-6.ipynb`

Running in sequence helps because concepts build on earlier phases.

## What You Practice In Each Notebook

- **Phase 1**: Tensor creation, shapes, element-wise ops, matrix multiplication, reshape/transpose, reproducible random tensors.
- **Phase 2**: `requires_grad`, `backward()`, manual parameter update with `torch.no_grad()`, optimizer-driven training.
- **Phase 3**: Synthetic 2D data, `DataLoader`, multi-layer MLP with `ReLU`, `Sigmoid`, `BCELoss`, test accuracy, loss plot.
- **Phase 4**: Synthetic tabular features (area, bedrooms, age), feature normalization, MLP regression, `MSELoss`, MSE/MAE, prediction scatter plot.
- **Phase 5**: Synthetic image generation, CNN (`Conv2d` + pooling), `CrossEntropyLoss`, classification accuracy, sample prediction visualization.
- **Phase 6**: Training a small classifier, saving/loading `state_dict`, validating loaded model predictions against original model.

## Notes

- The `torch_env/` folder is ignored by Git (see `.gitignore`) and can be used as your local virtual environment.
- The saved checkpoint (`pytorch_model_state.pth`) is a lightweight example artifact from Phase 6.

## Suggested Next Steps

1. Add a `requirements.txt` for one-command setup.
2. Add a single `run_all.py` script to execute all phases non-interactively.
3. Extend each phase with one challenge exercise and expected output.
