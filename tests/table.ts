import WASML from "../src/index"
import FoodGame from "./games/food"

// Uses the Q-Table method to teach a bot to find food.
const main = async () => {
  const game = new FoodGame(40)
  const wasml = new WASML()

  // 40x40 grid -> 1600 possible states.
  await wasml.table(5, 3, { epsilon: 0.025 })

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
