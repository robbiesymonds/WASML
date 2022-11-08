import WASML from ".."

const wasml = new WASML()

const main = async () => {
  await wasml.model(2, 2, { batchSize: 10, episodeSize: 10 })

  wasml.addLayers([
    {
      activation: "sigmoid",
      units: 2,
    },
  ])

  wasml.compile({ loss: "meanSquaredError" })

  // Trains a neural network to determine the largest number in a set of 2 numbers.
  let correct = 0
  for (let i = 0; i < 10000; i++) {
    let data: number[] = [Math.random(), Math.random()]
    let target: number[] = data[0] > data[1] ? [1, 0] : [0, 1]

    const result = wasml.predict(data)
    const reward =
      (result === 0 && data[0] > data[1]) || (result === 1 && data[0] < data[1]) ? 1 : -1
    wasml.reward(target, reward)
    if (reward > 0) correct++
    console.log(
      `Iteration: ${i + 1} | Test: [${data.join(", ")}] | Result: ${result} | Reward: ${reward}`
    )
  }
  console.log("Correct: ", (correct / 10000) * 100, "%")

  // Now attempt a prediction.
  console.log("Test: ", wasml.predict([0.5, 0.4]))
  console.log("Test: ", wasml.predict([0.1, 0.5]))
  console.log("Test: ", wasml.predict([0.1, 0.5]))
  console.log("Test: ", wasml.predict([0, 1]))
}

main()
