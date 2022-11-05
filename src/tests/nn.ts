import { NeuralNetwork } from "../network"

const NN = new NeuralNetwork(
  2,
  2,
  [
    {
      activation: "sigmoid",
      units: 4,
    },
    {
      activation: "sigmoid",
      units: 2,
    },
  ],
  {
    loss: "meanSquaredError",
    optimizer: "sgd",
  },
  0.1
)

// Trains a neural network to determine the largest number in a set of 2 numbers.
for (let i = 0; i < 10000; i++) {
  let data: number[] = [Math.random(), Math.random()]
  let target: number[] = data[0] > data[1] ? [1, 0] : [0, 1]

  const result = NN.forward(data)
  NN.backward(target)

  const loss = target.reduce((a, b, i) => a + (result[i] - b) ** 2, 0) / target.length
  console.log(`Iteration: ${i + 1} | Result: [${result.join(", ")}] | Loss: ${loss}`)
}

// Now attempt a prediction.
console.log("Test: ", NN.forward([10, 20]))
