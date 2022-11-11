import FoodGame from "./games/food"
import WASML from "../src/index"

// Uses the DQN to teach a bot to find food.
const main = async () => {
  const game = new FoodGame(40)
  const wasml = new WASML()

  await wasml.karparthy(5, 3, {
    epsilon: 0.01,
    alpha: 0.01,
    gamma: 0.975,
    episodeSize: 50,
    batchSize: 300,
    maxMemory: 1e6,
    epsilonDecay: 1e6,
  })

  // Import pre-trained model.
  const model = await fetch("./saves/food.json").then((res) => res.text())
  wasml.import(model)

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
