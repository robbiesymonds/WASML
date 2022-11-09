import FoodGame from "./games/food"
import { DQNSolver, DQNOpt, DQNEnv } from "reinforce-js"

// Uses the DQN to teach a bot to find food.
const main = async () => {
  const game = new FoodGame(40)

  const env = new DQNEnv(40, 40, 5, 3)
  const opt = new DQNOpt()
  opt.setTrainingMode(true)
  opt.setNumberOfHiddenUnits([64])
  opt.setEpsilonDecay(0.2, 0.025, 1e6)
  opt.setEpsilon(0.2)
  opt.setGamma(0.975)
  opt.setAlpha(0.01)
  opt.setLossClipping(true)
  opt.setRewardClipping(false)
  opt.setExperienceSize(1e6)
  opt.setReplayInterval(350)
  opt.setReplaySteps(250)

  const wasml = new DQNSolver(env, opt)

  // Runs the game loop.
  // - The state consists of 5 values: the movement direction, and 4 values indicating whether the food is above, below, left, or right of the bot.
  game.loop((state) => {
    // Get the action.
    const action = wasml.decide(state)
    console.log(state)

    // Perform the action.
    game.move(action)

    // Get the reward.
    const reward = game.reward()
    console.log(reward)

    // Render new state.
    game.render()

    // Train the AI.
    wasml.learn(reward)
  })
}

main()
