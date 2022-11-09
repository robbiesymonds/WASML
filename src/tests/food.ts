import WASML from "../index"
import FoodGame from "./games/food"

const main = async () => {
  const game = new FoodGame()
  const wasml = new WASML()

  // Setup the AI.
  await wasml.model(6, 3, {
    alpha: 0.1,
    gamma: 0.85,
    epsilon: 0.1,
    batchSize: 500,
    maxMemory: 50000,
    episodeSize: 100,
  })
  wasml.addLayers([
    { units: 80, activation: "sigmoid" },
    { units: 3, activation: "linear" },
  ])
  wasml.compile({ loss: "meanSquaredError" })

  game.loop((state) => {
    // Get the action from the AI.
    const action = wasml.predict(state)

    // Perform the action.
    game.move(action)

    // Get the reward.
    const reward = game.reward()
    console.log(reward)

    // Get the next state.
    const next = game.state()
    console.log(next)

    // Train the AI.
    wasml.reward(reward, next)

    // Render new state.
    game.render()
  })
}

main()
