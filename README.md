# MNIST Digit Classifier in C

A small neural-network project written in C that trains a model to recognize handwritten digits from the MNIST dataset. It includes a Python script to download and prepare MNIST, a C implementation of the model and training loop, persistent model checkpoints, and commands to inspect predictions and evaluate accuracy.

The network takes a 28 x 28 grayscale image (784 values) and predicts one of the digits 0 through 9. The C program can print test images in the terminal, along with their actual labels and predicted digits.

## Requirements

- GCC with C99 support (MinGW-w64 GCC works on Windows)
- Python 3
- NumPy
- An internet connection the first time you prepare the dataset

## Setup

Clone the repository and open a terminal in its folder:

```powershell
git clone https://github.com/5umit1711/MNIST_DigitClassifier_in_C.git
cd MNIST_DigitClassifier_in_C
```

Install NumPy and prepare the dataset. The script downloads the four MNIST files and creates the `.mat` files used by the C program:

```powershell
python -m pip install numpy
python mnist.py
```

On Windows, if `python` is not recognized, use the Python launcher:

```powershell
py -m pip install numpy
py mnist.py
```

Compile the program with GCC:

```powershell
gcc -std=c99 -O2 main.c -lm -o a.exe
```

On Linux or macOS, use `python3` and compile to `a`:

```sh
python3 -m pip install numpy
python3 mnist.py
gcc -std=c99 -O2 main.c -lm -o a
```

Run `./a` in place of `.\a.exe` in the commands below.

## Train

Train a new model and save its parameters to `mnist_model.bin`:

```powershell
.\a.exe train
```

Training uses 10 epochs and reports progress and test-set accuracy after each epoch. The checkpoint is saved in the current directory. Running `train` again starts training from newly initialized weights and replaces the checkpoint when training finishes.

If no checkpoint exists, running the program without a command will train and save one automatically:

```powershell
.\a.exe
```

## Check predictions

Display images from the MNIST test set in the terminal and show the actual label, predicted digit, and prediction confidence:

```powershell
.\a.exe predict
```

By default, this displays the first 10 test images. To display specific test images, pass one or more indexes from `0` to `9999`:

```powershell
.\a.exe predict 0 1 42
```

The `actual` value is the dataset's label; `predicted` is the model's answer. A prediction marked `INCORRECT` does not match the label.

## Evaluate accuracy

Evaluate against the full 10,000-image MNIST test set and show both overall and per-digit accuracy:

```powershell
.\a.exe test
```

To evaluate only the first `count` test images, pass a count from `1` to `10000`:

```powershell
.\a.exe test 100
```

The `test` and `predict` commands load `mnist_model.bin`; if it is missing, train the model first.

## Saved and generated files

The dataset archives (`*.gz`), prepared arrays (`*.mat`), model checkpoint (`*.bin`), and compiled executables (`*.exe`) are generated locally and excluded from Git. Run `python mnist.py` to prepare the dataset again, and `.\a.exe train` to create a checkpoint.
