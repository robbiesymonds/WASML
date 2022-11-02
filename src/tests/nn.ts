import { NeuralNetwork } from "../network"

const NN = new NeuralNetwork(
  2,
  2,
  [
    {
      activation: "relu",
      units: 2,
    },
  ],
  {
    loss: "meanSquaredError",
    optimizer: "sgd",
  },
  0.01
)

let data: number[] = [3, 3]
let target: number[] = [4, 16]

for (let i = 0; i < 100; i++) {
  const result = NN.forward(data)
  console.log(result)

  const loss = target.reduce((a, b, i) => a + (result[i] - b) ** 2, 0)
  console.log(loss)

  NN.fit(target)

  console.log(`Iteration: ${i + 1} | Result: ${result} | Loss: ${loss}`)
}
