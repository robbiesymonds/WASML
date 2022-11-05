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

WASML can be imported as both an ES and CommonJS module. The syntax takes heavy inspiration from TensorflowJS, so it should familiar to those with some prior experience. The following example demontrates the basic usage.

#### Basic Usage
```jsx
import WASML from "wasml"

const wasml = new WASML()

// Create a model with 16 inputs and 8 action states.
await wasml.model(16, 8, { replay: true }) // See below for full configuration options.

// Add two hidden layers.
wasml.addLayers([
  { type: "convolution", shape: 12, activation: "sigmoid" },
  { type: "dense", shape: 8, activation: "relu" },
])

// Compile the model.
wasml.compile({ loss: "meanSquaredError", optimizer: "sgd" })

// Array of a hundred empty samples. (Obviously have real data)
// - It is not neccessary to pre-train the model, but can be useful.
const inputs = Array(100).from(new Float32Array(16))
const outputs = Array(100).from(new Float32Array(8))
wasml.train(inputs, outputs)

// Predict the optimal action.
const input = new Float32Array(16)
const result = wasml.predict(input)

// [?] Do something with the action.

// Reward the model.
wasml.reward(1.0)

```

#### Import/Export
```jsx
import WASML from "wasml"

const wasml = new WASML()

// Load an exported model and restore the memory.
const model = readFileSync('export.json')
await wasml.load(model)

const input = new Float32Array(16)
const result = wasml.predict(input)
wasm.reward(-1.0)

// Get the memory of the changed model in JSON form.
const json = await wasml.save()
```

## ⚙️ Configuration
Text