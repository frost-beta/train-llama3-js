#!/usr/bin/env node

import fs from 'node:fs'
import path from 'node:path'
import tokenizer from 'llama3-tokenizer-js'
import parquet from "@dsnp/parquetjs"
import {core as mx, optimizers as optim, nn, utils} from '@frost-beta/mlx'

import Model from './model.js'

if (process.argv.length < 3) {
  console.error('Usage: train.js /path/to/train-*.parquet')
  process.exit(0)
}

// Hyperparameters.
const contextSize = 128

// Traning configs.
const epochs = 1
const batchSize = 1
const learningRate = 1e-3

main()

async function main() {
  // Read the config.json from current dir and create the model from it.
  const config = JSON.parse(fs.readFileSync('config.json'))
  const model = new Model(config)

  // const weightsFile = 'weights.safetensors'
  // if (fs.existsSync(weightsFile)) {
  //   model.loadWeights(weightsFile)
  //   console.log(model.parameters())
  // }

  // Calculate how many parameters the model has.
  let nparams = 0
  for (const [k, x] of utils.treeFlatten(model.parameters())) {
    if (!k.includes('embedTokens'))
      nparams += x.size
  }
  console.log(`Training Llama3 with ${(nparams / 1024 ** 2).toFixed(1)}M parameters.`)

  // Preprare utils for doing gradient descent.
  const lossAndGradFunction = nn.valueAndGrad(model, lossFunction)
  const optimizer = new optim.AdamW(learningRate)

  // Read batches from the datasets passed via command line.
  const files = process.argv.slice(2)
  const reportPerIter = 10
  let losses = []
  for (let e = 0, iterations = 1, start = Date.now(); e < epochs; ++e) {
    for await (const [x, y] of iterateBatches(files, batchSize, contextSize)) {
      // Use mx.tidy to free all the intermediate tensors immediately.
      mx.tidy(() => {
        // Compute loss and gradients, then update the model.
        const [loss, grads] = lossAndGradFunction(model, x, y)
        optimizer.update(model, grads)
        mx.eval(model.state, optimizer.state)
        losses.push(loss.item())
        // Keep the states of model and optimizer from getting freed.
        return [model.state, optimizer.state]
      })
      mx.dispose([x, y])
      // Report updates.
      if (++iterations % reportPerIter === 0) {
        const stop = Date.now()
        const trainLoss = mean(losses)
        console.log(`Iter ${iterations}:`,
                    `Train loss ${trainLoss.toFixed(3)},`,
                    `It/sec ${(reportPerIter / (stop - start) * 1000).toFixed(3)}.`)
        start = Date.now()
        losses = []
      }
    }
  }

  // console.log('Saving weights...')
  // model.saveWeights(weightsFile)
}

// Read datasets from |files|, and generate batches of [inputs, targets].
async function* iterateBatches(files, batchSize, contextSize) {
  let inputBatch = []
  let outputBatch = []
  for (const f of files) {
    // Read the dataset.
    const reader = await parquet.ParquetReader.openFile(f)
    const cursor = reader.getCursor()
    let record
    while (record = await cursor.next()) {
      // Convert text to tokens.
      const tokens = tokenizer.encode(record.text)
      // Generate batches from the tokens.
      for (let i = 0; i < tokens.length - 1; i += contextSize) {
        const length = Math.min(contextSize, tokens.length - i - 1);
        inputBatch.push(tokens.slice(i, i + length))
        outputBatch.push(tokens.slice(i + 1, i + 1 + length))
      }
      // Yield batches with each batch of |batchSize|.
      while (inputBatch.length >= batchSize) {
        const x = inputBatch.splice(0, batchSize)
        const y = outputBatch.splice(0, batchSize)
        yield [ mx.array(x, mx.int32), mx.array(y, mx.int32) ]
      }
    }
    await reader.close()
  }
}

// Calculate the loss by 1) running the model with the inputs, and 2) then using
// cross entropy function to get the loss between the results and targets.
function lossFunction(model, x, y) {
  const [logits, cache] = model.forward(x)
  const losses = nn.losses.crossEntropy(logits, y)
  return mx.mean(losses)
}

// Compute the mean value of an array.
function mean(array) {
  if (array.length == 0)
    return 0
  return array.reduce((a, b) => a + b) / array.length
}
