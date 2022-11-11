import WASML from "../src/index"
import FoodGame from "./games/food"

// Uses the DQN to teach a bot to find food.
const main = async () => {
  const game = new FoodGame(40)
  const wasml = new WASML()

  // 1 hidden layer with 60 neurons + 1 output layer.
  await wasml.model(5, 3, {
    epsilon: 0.15,
    alpha: 0.001,
    gamma: 0.975,
    episodeSize: 50,
    batchSize: 300,
    maxMemory: 1e6,
    epsilonDecay: 1e6,
  })
  wasml.addLayers([
    {
      units: 64,
      activation: "sigmoid",
    },
    { units: 3, activation: "linear" },
  ])
  wasml.compile({ loss: "meanAbsoluteError" })

  // Runs the game loop.
  // - The state consists of 5 values: the movement direction, and 4 values indicating whether the food is above, below, left, or right of the bot.
  game.loop((state) => {
    // Get the action.
    const action = wasml.predict(state)

    // Perform the action.
    game.move(action)

    // Get the reward.
    const reward = game.reward()

    // Render new state.
    game.render()

    // Get the next state.
    const next = game.state()

    // Train the AI.
    wasml.reward(reward, next)
  })
}

main()
