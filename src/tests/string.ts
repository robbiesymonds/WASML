// import WASML from "../.."
import WASML from "../index"

const wasml = new WASML()

let BEST: number = Infinity
const TARGET: string = "hello"

const main = async () => {
  await wasml.model(5, 3)
  wasml.compile({ loss: "meanSquaredError", optimizer: "sgd" })

  const run = () => {
    let idx = 0
    let iter = 0
    let STATE: string = "aaaaa"

    // Gets substring of the state string.
    const s = (a: number, b?: number) => STATE.substring(a, b)

    // Run until we find the target.
    while (true) {
      const input = new Float32Array(STATE.split("").map((char) => char.charCodeAt(0) - 97))
      console.log(input, input.length)
      const action = wasml.predict(input)

      console.log("Action:", action)

      switch (action) {
        case 0: {
          // Decrease the character at the current index.
          const v = Math.max(97, Math.min(122, STATE.charCodeAt(idx) - 1))
          STATE = `${s(0, idx)}${String.fromCharCode(v)}${s(idx + 1)}`
          break
        }

        case 1: {
          // Increase the character at the current index.
          const v = Math.max(97, Math.min(122, STATE.charCodeAt(idx) + 1))
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
      const new_state = new Float32Array(STATE.split("").map((char) => char.charCodeAt(0) - 97))

      if (STATE === TARGET) {
        console.log("Big Reward!")
        if (iter < BEST) BEST = iter
        wasml.reward(new_state, 99)
        break
      } else {
        // Penalty for incorrect final string.
        if (idx === STATE.length - 1) {
          console.log("Big Penalty!")
          wasml.reward(new_state, -10.0)
          break
        } else {
          // Reward based on whether agent moved closer to target.
          const reward =
            action === 0 && STATE.charCodeAt(idx) > TARGET.charCodeAt(idx)
              ? 1.0
              : action === 1 && STATE.charCodeAt(idx) < TARGET.charCodeAt(idx)
              ? 1.0
              : action === 2 && STATE.charCodeAt(idx) === TARGET.charCodeAt(idx)
              ? 2.0
              : -1.0
          console.log("Reward:", reward)
          wasml.reward(new_state, reward)
        }
      }
    }

    console.log("Best:", BEST)
  }

  run()
}

main()
