import WASML from "../index"

const wasml = new WASML()

let TARGET: number = 12

const main = async () => {
  await wasml.model(1, 3, {
    batchSize: 1,
    episodeSize: 20,
    maxMemory: 1000,
    epsilon: 0.05,
    gamma: 0.9,
  })
  wasml.addLayers([{ units: 3, activation: "sigmoid" }])
  wasml.compile({ loss: "meanSquaredError" })

  const run = () => {
    let STATE: number = 6
    let s: number = 0

    while (s < 50) {
      s++

      const action = wasml.predict([STATE])

      switch (action) {
        case 0: {
          STATE -= 1
          break
        }
        case 1: {
          STATE += 1
          break
        }
      }

      const reward =
        (action === 0 && TARGET < STATE) || (action === 1 && TARGET > STATE)
          ? 1
          : action === 2 && TARGET === STATE
          ? 2
          : action == 2 && TARGET !== STATE
          ? -2
          : -1

      // Train the model.
      console.log("New State: ", STATE, "| Action: ", action, "| Reward:", reward)
      wasml.reward([STATE], reward)
    }

    if (TARGET === STATE) console.log(`Reached target in ${s} steps.`)
  }

  for (let i = 0; i < 1; i++) {
    run()
  }

  console.log("Tests:")
  console.log("What to do at x=5?", wasml.predict([5]))
  console.log("What to do at x=14?", wasml.predict([14]))
  console.log("What to do at x=12?", wasml.predict([12]))
}

await main()
