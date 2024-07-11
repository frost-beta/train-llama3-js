# Train Llama3.js

This repo hosts educational scripts showing how to train a Llama3 model with
public datasets on HuggingFace.

It is recommended to take a look at the [Train text generation model with
JavaScript](https://github.com/frost-beta/train-model-with-js) tutorial first.

## Platform

Only Macs with Apple Silicon are supported.

## Usage

Install dependencies:

```sh
npm install
```

Download a dataset:

```sh
npm install -g @frost-beta/huggingface
huggingface download --to dataset-tinystories --filter=*.parquet datasets/roneneldan/TinyStories
```

Train with it:

```sh
node train.js dataset-tinystories/data/train-*.parquet
```
