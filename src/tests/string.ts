import WASML from "../index"

const wasml = new WASML()

let BEST: number = Infinity
const TARGET: string = "abc"

const main = async () => {
  await wasml.model(3, 3, { batchSize: 2, episodeSize: 10, maxMemory: 1000 })
  wasml.addLayers([
    { units: 6, activation: "sigmoid" },
    { units: 3, activation: "sigmoid" },
  ])
  wasml.compile({ loss: "meanSquaredError" })

  const run = () => {
    let idx = 0
    let iter = 0
    let STATE: string = "aaa"

    // Gets substring of the state string.
    const s = (a: number, b?: number) => STATE.substring(a, b)

    // Run until we find the target.
    while (true) {
      const input = STATE.split("").map((char) => char.charCodeAt(0) - 97)
      console.log("Input: ", input)
      const action = wasml.predict(input)

      console.log("Action: ", action)

      switch (action) {
        case 0: {
          // Decrease the character at the current index.
          const c = STATE.charCodeAt(idx) - 1
          const v = c < 97 ? 122 : c
          STATE = `${s(0, idx)}${String.fromCharCode(v)}${s(idx + 1)}`
          break
        }

        case 1: {
          // Increase the character at the current index.
          const c = STATE.charCodeAt(idx) + 1
          const v = c > 122 ? 97 : c
          STATE = `${s(0, idx)}${String.fromCharCode(v)}${s(idx + 1)}`
          break
        }

        case 2: {
          // Move onto the next index.
          idx++
        }
      }

      // Increase iteration count.
      iter++

      // Big reward if we find the target.
      const new_state = STATE.split("").map((char) => char.charCodeAt(0) - 97)

      if (STATE === TARGET) {
        if (iter < BEST) BEST = iter
        wasml.reward(new_state, 2.0)
        break
      } else {
        // Penalty for incorrect final string.
        if (idx === STATE.length) {
          wasml.reward(new_state, -2.0)
          break
        } else {
          // Reward based on whether agent moved closer to target.
          const reward =
            (action === 0 && STATE.charCodeAt(idx) > TARGET.charCodeAt(idx)) ||
            (action === 1 && STATE.charCodeAt(idx) < TARGET.charCodeAt(idx))
              ? 1.0
              : action === 2 && STATE.charCodeAt(idx) === TARGET.charCodeAt(idx)
              ? 1.0
              : -1.0
          wasml.reward(new_state, reward)
        }
      }
    }

    console.log("Best:", BEST, "Final State: ", STATE)
  }

  for (let i = 0; i < 3; i++) {
    run()
  }
}

main()
