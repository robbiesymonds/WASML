import { Tensor } from "../math"
import { Layer, NeuralNetwork } from "../network"

// One hidden layer and one output layer.
const LAYERS: Layer[] = [
  {
    activation: "sigmoid",
    units: 8,
  },
  {
    activation: "sigmoid",
    units: 2,
  },
]

// Create a new neural network.
const NN = new NeuralNetwork(2, 2, LAYERS, { loss: "meanSquaredError" }, 0.1)

// Trains a neural network to determine the largest number in a set of 2 numbers.
let correct: number = 0
for (let i = 0; i < 10000; i++) {
  let data: number[] = [Math.random(), Math.random()]
  let target: number[] = data[0] > data[1] ? [1, 0] : [0, 1]

  // Perform forward prediction and then backpropagate.
  const result = NN.forward(data)
  NN.backward(target)

  // Check if the network was correct.
  if (Tensor.argmax(result) === Tensor.argmax(target)) correct++
  console.log(`Iteration: ${i + 1} | Result: [${result.join(", ")}]`)
}

console.log(`Total Accuracy: ${(correct / 10000) * 100}%`)

// Now attempt some static predictions.
const TESTS = [
  [10, 20],
  [500, 1],
  [0.7, 0.99],
]
TESTS.forEach((t) => {
  const result = NN.forward(t)
  const max = t[Tensor.argmax(result)]
  console.log(
    `Test: max(${t.join(", ")}) | Guess: ${max} | Confidence: ${Math.max(...result) * 100}%`
  )
})
