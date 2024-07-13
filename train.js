#!/usr/bin/env node

import fs from 'node:fs'
import path from 'node:path'
import prettyMilliseconds from 'pretty-ms'
import {ParquetReader} from '@dsnp/parquetjs'
import {fromPreTrained} from '@lenml/tokenizer-llama3'
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
const batchSize = 128 + 64
const learningRate = 1e-4

main()

async function main() {
  // Read the config.json from current dir and create the model from it.
  const config = JSON.parse(fs.readFileSync('config.json'))
  const model = new Model(config)

  // Get the tokenizer of Llama3.
  const tokenizer = fromPreTrained()

  // Calculate how many parameters the model has.
  let nparams = 0
  for (const [k, x] of utils.treeFlatten(model.parameters())) {
    if (!k.includes('embedTokens'))
      nparams += x.size
  }
  console.log(`Training Llama3 with ${(nparams / 1024 ** 2).toFixed(1)}M parameters.`)

  // Command line flags.
  const files = process.argv.slice(2)
  const totalRows = await getRowCount(files)
  const reportPerIter = Math.max(Math.floor(32 / batchSize * 10), 1)
  console.log('Total rows of data to train:', totalRows)

  // Preprare utils for doing gradient descent.
  const lossAndGradFunction = nn.valueAndGrad(model, lossFunction)
  const optimizer = new optim.AdamW(learningRate)

  // Read batches from the datasets.
  let lastRow = 0
  let losses = []
  for (let e = 0, iterations = 0, start = Date.now(); e < epochs; ++e) {
    for await (const [row, x, y] of iterateBatches(files, tokenizer, contextSize, batchSize)) {
      // Use mx.tidy to free all the intermediate tensors immediately.
      mx.tidy(() => {
        // Compute loss and gradients, then update the model.
        const [loss, grads] = lossAndGradFunction(model, mx.array(x, mx.int32), mx.array(y, mx.int32))
        optimizer.update(model, grads)
        mx.eval(model.state, optimizer.state)
        losses.push(loss.item())
        // Keep the states of model and optimizer from getting freed.
        return [model.state, optimizer.state]
      })
      // Report updates.
      if (++iterations % reportPerIter === 0) {
        const stop = Date.now()
        const trainLoss = mean(losses)
        const eta = (totalRows / (row - lastRow)) * (stop - start)
        console.log(`Iter ${iterations}`,
                    `(${(100 * row / totalRows).toFixed(1)}%):`,
                    `Train loss ${trainLoss.toFixed(2)},`,
                    `It/sec ${(reportPerIter / (stop - start) * 1000).toFixed(2)},`,
                    `ETA ${prettyMilliseconds(eta, {compact: true})}.`)
        start = Date.now()
        losses = []
        lastRow = row
      }
      // Check for leaks.
      if (false && iterations % 100 === 0) {
        console.log(`MLX RAM ${(mx.metal.getActiveMemory() / 1024 ** 2).toFixed(1)}M,`,
                    `Cache ${(mx.metal.getCacheMemory() / 1024 ** 2).toFixed(1)}M,`,
                    `JS Objects ${mx.getWrappersCount()}.`)
      }
    }
  }

  // Save weights on exit.
  console.log('Saving weights...')
  model.saveWeights(weightsFile)
}

// Return the total number of rows.
async function getRowCount(files) {
  let count = 0
  for (const f of files) {
    const reader = await ParquetReader.openFile(f)
    count += parseInt(reader.getRowCount())
    await reader.close()
  }
  return count
}

// Read datasets from |files|, and generate batches of [inputs, targets].
async function* iterateBatches(files, tokenizer, contextSize, batchSize) {
  const eosToken = tokenizer.encode(tokenizer.getToken('eos_token'))[0]
  let row = 0
  let inputBatch = []
  let outputBatch = []
  for (const f of files) {
    // Read the dataset.
    const reader = await ParquetReader.openFile(f)
    const cursor = reader.getCursor()
    let record
    while (record = await cursor.next()) {
      ++row
      // Convert text to tokens.
      const tokens = tokenizer.encode(record.text)
      // Generate batches from the tokens.
      for (let i = 0; i < tokens.length - 1; i += contextSize) {
        const length = Math.min(contextSize, tokens.length - i - 1)
        // If the batch's length is less than contextSize, fill it with EOS.
        let paddings = []
        if (length < contextSize)
          paddings = new Array(contextSize - length).fill(eosToken)
        inputBatch.push(tokens.slice(i, i + length).concat(paddings))
        outputBatch.push(tokens.slice(i + 1, i + 1 + length).concat(paddings))
      }
      // Yield batches with each batch of |batchSize|.
      while (inputBatch.length >= batchSize) {
        yield [ row, inputBatch.splice(0, batchSize), outputBatch.splice(0, batchSize) ]
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
