# Simplify ONNX

This project provides a Python script for simplifying ONNX models. The script applies a series of modifiers to the model, such as simplification, reshape modification, and slice merging, to produce a simplified version of the model.

## Installation

To use this script, you'll need to have Python 3 installed on your system. You can install the required Python packages by running the following command:

```shell
pip install -r requirements.txt
```

It is highly recommended to install numpy with conda first.
```shell
conda install numpy -y
```

## Usage

To use the script, run the following command:

```shell
python -m streamlit run simplify_onnx.py
```

or

```shell
streamlit run simplify_onnx.py
```

This will launch a web interface where you can select an ONNX file and apply modifiers to it. Once you've selected your modifiers, click the "Start" button to run the script.
You can visit it from http://localhost:8501

## Modifiers

The following modifiers are available:

- simplify
    simplify the model with onnx-simplifier
- modify_reshape
    modifies reshape nodes to use constant values and keep batch -1
- replace_squeeze_and_unsqueeze
    replaces squeeze and unsqueeze nodes with reshape nodes
- merge_slice
    merges slice nodes with single split node
- reshape_output
    reshapes the output to 4D

You can add modifiers to the script by clicking the "Add" button next to the modifier name. You can also clear all modifiers by clicking the "Clear modifiers" button.

## Intermediate Results

You can choose to save intermediate results by checking the "Save intermediate results" checkbox. This will save the model after each modifier has been applied.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
