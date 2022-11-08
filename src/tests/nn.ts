import { Tensor } from "../math"
import { NeuralNetwork } from "../network"

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
let correct: number = 0
for (let i = 0; i < 10000; i++) {
  let data: number[] = [Math.random(), Math.random()]
  let target: number[] = data[0] > data[1] ? [1, 0] : [0, 1]

  const result = NN.forward(data)
  if (Tensor.argmax(result) === Tensor.argmax(target)) correct++
  NN.backward(target)

  const loss = target.reduce((a, b, i) => a + (result[i] - b) ** 2, 0) / target.length
  console.log(`Iteration: ${i + 1} | Result: [${result.join(", ")}] | Loss: ${loss}`)
}

console.log(`Accuracy: ${(correct / 10000) * 100}%`)

// Now attempt some static predictions.
console.log("Test 1: ", NN.forward([10, 20]))
console.log("Test 2: ", NN.forward([500, 1]))
console.log("Test 3: ", NN.forward([0.7, 0.99]))
