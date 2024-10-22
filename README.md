# Train Llama3 with JavaScript

This repo hosts educational scripts for traning a Llama3 model with parquet
datasets.

This is a slightly advanced example than the [train-model-with-js](https://github.com/frost-beta/train-model-with-js)
repo with a few new things added:

1. Model is trained on a large parquet dataset for a long time, instead of a
   single text file for a few minutes.
2. The tokenizer of Llama3 is used.
3. Generate Llama3 weights which can be loaded by any inference engine.
4. There is an estimation of how long the training will take.

## Platform

Only Macs with Apple Silicon are supported.

## Preparations

First clone this repo and install dependencies:

```sh
git clone https://github.com/frost-beta/train-llama3-js.git
cd train-llama3-js
npm install
```

Then download the dataset for training. You can use any parquet dataset on
[HuggingFace](https://huggingface.co/datasets?modality=modality:text&format=format:parquet&sort=trending),
for beginners, I suggest using the synthetic TinyStories dataset:

```sh
npm install -g @frost-beta/huggingface
huggingface download datasets/Chat-Error/tinystories-gpt4
```

The model's configurations are coded in the `config.json` file, which you can
change to make it a smaller or a bigger model.

```json
{
  "model_type": "llama",
  "hidden_size": 128,
  "num_hidden_layers": 8,
  "intermediate_size": 32,
  "num_attention_heads": 4,
  "rms_norm_eps": 1e-06,
  "vocab_size": 128256,
  "num_key_value_heads": 4
}
```

The traning script `train.js` includes some hyperparameters which you might want
to tune, currently it is set so a M3 Max 32GB machine trains with first 300k
entries of the TinyStories dataset for about 1 hour.

```js
// Traning configs.
const epochs = 1
const batchSize = 32
const learningRate = 1e-4
// Max rows of date to train, set to Infinity to train everything.
const maxRows = 300 * 1000
```

For machines with smaller RAM, you should change `batchSize` to a smaller size
like 16, which will take more time to train but requires much less RAM. And by
changing `maxRows` you can control how long the training will be.

## Training

To start training, just pass the paths of parquet files to the `train.js`
script:

```sh
node train.js tinystories-gpt4/train.parquet
```

It will output the progress of the training like this:

```
Iter 10 (0.3%): Train loss 9.49, It/sec 0.99, ETA 54m.
Iter 11 (0.3%): Train loss 9.17, It/sec 0.99, ETA 52m.
Iter 12 (0.4%): Train loss 9.30, It/sec 0.99, ETA 58m.
Iter 13 (0.4%): Train loss 9.04, It/sec 1.00, ETA 55m.
Iter 14 (0.4%): Train loss 8.84, It/sec 0.99, ETA 53m.
```

After the training is done, a `weights.safetensors` file will be written. By
providing the `config.json` and `weights.safetensors` files, you can load your
own model with any inference that supports Llama3:

```sh
npm install -g llama3
llama3-generate . 'Once upon a time'
```

## What's next

* [train-japanese-llama3-js](https://github.com/frost-beta/train-japanese-llama3-js) -
  Train a Japanese language model.
* [fine-tune-decoder-js](https://github.com/frost-beta/fine-tune-decoder-js) -
  Fine-tune a decoder-only model.

## License

Public domain.
