<img src="tests/favicon.ico" width="100"  />

# WASML

WebAssembly-powered reinforcement learning library written in Rust and TypeScript. 

## 🚀 Getting Started
WASML is available as an NPM package, simply install with the package manager of your choice.
```sh
# With yarn
yarn add wasml

# With npm
npm install wasml --save
```

## 💾 Usage

WASML can be imported as both an ES and CommonJS module. The syntax takes heavy inspiration from TensorflowJS, so it should familiar to those with some prior experience. The following examples demontrates the basic usage (see `/src/tests/` for more).

#### Basic Usage
```jsx
import WASML from "wasml"

const wasml = new WASML()

// Create a model with 16 inputs and 3 action states.
await wasml.model(16, 3) // See below for full configuration options.

// Using `wasml.table(16, 3)` instead will solve this game far quicker!
// * Tabular optimisation becomes less feasible as state space grows (only 40x40=1600 states here)

// Add two hidden layers.
wasml.addLayers([
  { units: 32, activation: "sigmoid" },
  { units: 8, activation: "linear" },
])

// Compile the model.
wasml.compile({ loss: "meanSquaredError" })

// Array of a hundred empty samples.
// - It is not neccessary to pre-train the model, but can be useful.
const inputs = Array(100).from(Array.from({ length: 16 }, () => Math.random()))
const outputs = Array(100).from([1, 0, 0])
wasml.train(inputs, outputs)

// Predict the optimal action.
const input = Array.from({ length: 16 }, () => Math.random())
const result = wasml.predict(input)

// [?] Do something with the action.

// Reward the model.
wasml.reward(10.0)
```

#### Import/Export
```jsx
import WASML from "wasml"

const wasml = new WASML()

// Load an exported model and restore the memory.
const model = await fetch('export.json').then(res => res.text())
wasml.import(model)

// Get the memory of the changed model in JSON form.
const json = wasml.export()
```

#### Custom Neural Network
```jsx
import { NeuralNetwork } from "wasml/network"

// Utilise the underlying NN.
const NN = new NeuralNetwork(
  2,
  2,
  [
    {
      activation: "sigmoid",
      units: 8,
    },
    {
      activation: "sigmoid",
      units: 2,
    },
  ],
  {
    loss: "meanSquaredError",
  },
  0.1
)

// Trains a neural network to determine the largest number in a set of 2 numbers.
for (let i = 0; i < 10000; i++) {
  let data: number[] = [Math.random(), Math.random()]
  let target: number[] = data[0] > data[1] ? [1, 0] : [0, 1]

  const result = NN.forward(data)
  NN.backward(target)
}

// Now attempt some static predictions.
console.log("Test 1: ", NN.forward([10, 20]))
console.log("Test 2: ", NN.forward([500, 1]))
console.log("Test 3: ", NN.forward([0.7, 0.99]))
```

## ⚙️ Configuration
The following are collection of optional parameters that can be passed as options to WASML.
| Name | Type | Default | Description |
|------|------|---------|-------------|
`alpha`|number| 0.1 | The learning rate of the model. |
`gamma`|number| 0.95 | The reward discount factor, usually in range (0, 1).
`epsilon`|number| 0.1 | The probability of performing a random action.
`maxMemory`|number| 1000 | The size of the experience replay memory.
`batchSize`|number| 100 | The number of experiences to sample each iteration.
`episodeSize`|number| 50 | The number of iterations before updating target network.
`epsilonDecay`|number| 1000000 | The number of iterations over which epsilon tends to zero.