import init from "../dist/wasml"
import { Memory } from "./memory"
import { Layer, NeuralNetwork } from "./network"
import { Tensor } from "./math"
import { Table } from "./table"
import { DQNAgent } from "./lib/karparthy.js"

export interface BaseOptions {
  alpha: number
  gamma: number
  epsilon: number
  epsilonDecay: number
}

interface ModelOptions extends BaseOptions {
  maxMemory: number
  batchSize: number
  episodeSize: number
}

export interface CompileOptions {
  loss: "meanSquaredError" | "meanAbsoluteError"
}

// The default configurations for the model.
const DEFAULT_MODEL_OPTIONS: ModelOptions = {
  alpha: 0.1,
  gamma: 0.95,
  epsilon: 0.1,
  maxMemory: 1000,
  batchSize: 100,
  episodeSize: 50,
  epsilonDecay: 1e6,
}

const DEFAULT_COMPILE_OPTIONS: CompileOptions = {
  loss: "meanSquaredError",
}

enum Mode {
  TABLE,
  MODEL,
  KARPARTHY,
}

export default class WASML {
  private init: boolean = false
  private states: number = Infinity
  private actions: number = Infinity
  private options: ModelOptions = DEFAULT_MODEL_OPTIONS
  private config: CompileOptions = DEFAULT_COMPILE_OPTIONS
  private layers: Layer[] = []
  private mode!: Mode

  private DQN!: NeuralNetwork
  private Target!: NeuralNetwork
  private Memory!: Memory
  private QTable!: Table
  private KARPATHY!: DQNAgent

  private last_state!: number[]
  private last_action!: number
  private episode: number = 0
  private epsilon: number = 0

  /**
   * Loads the WASM binary into the browser.
   * @returns {Promise<void>} - A promise that resolves when the WASM binary has been loaded.
   */
  static async load(): Promise<void> {
    await init()
  }

  /**
   * Generates a model with the given options.
   * @param {number} states - The size of the input vector space for the model.
   * @param {number} actions - The number of actions the model can take.
   * @param {ModelOptions} options - The options for the model.
   */
  async model(states: number, actions: number, options?: Partial<ModelOptions>): Promise<void> {
    this.states = states
    this.actions = actions
    this.options = { ...DEFAULT_MODEL_OPTIONS, ...options }
    this.epsilon = this.options.epsilon

    await WASML.load().then(() => {
      this.mode = Mode.MODEL
    })
  }

  /**
   * Uses Andrej Karpathy's method to train the model.
   * @note For some reason, the `.model()` method trains significantly slower than this method, so keeping it for now.
   * @param {number} states - The size of the input vector space for the model.
   * @param {number} actions - The number of actions the model can take.
   * @param {ModelOptions} options - The options for the model.
   * @deprecated This method will be removed in a future release.
   */
  async karparthy(states: number, actions: number, options?: Partial<ModelOptions>): Promise<void> {
    this.states = states
    this.actions = actions
    this.options = { ...DEFAULT_MODEL_OPTIONS, ...options }
    this.epsilon = this.options.epsilon

    await WASML.load().then(() => {
      this.mode = Mode.KARPARTHY
      this.KARPATHY = new DQNAgent(
        { getNumStates: () => states, getMaxNumActions: () => actions },
        this.options
      )
    })
  }

  /**
   * Generates a Q-Learning table with the given options.
   * @param {number} states - The size of the input vector space for the table.
   * @param {number} actions - The number of actions the table can take.
   * @param {BaseOptions} options - The options for the table.
   */
  async table(states: number, actions: number, options?: Partial<BaseOptions>): Promise<void> {
    this.states = states
    this.actions = actions
    this.options = { ...DEFAULT_MODEL_OPTIONS, ...options }
    this.epsilon = this.options.epsilon
    this.QTable = new Table(states, actions)

    await WASML.load().then(() => {
      this.mode = Mode.TABLE
    })
  }

  /**
   * Checks if the model has been initialised.
   * @returns {boolean} - Whether the model has been initialised.
   */
  private initialised(): void {
    if (!this.init && this.mode === Mode.MODEL) {
      throw new Error("A call to `.compile()` is required before interaction is performed!")
    }
  }

  /**
   * Adds the specified layers to the neural network.
   * @param {Layer[]} layers - The array of layers to add to the neural network.
   */
  addLayers(layers: Layer[]): void {
    if (this.mode === Mode.TABLE || Mode.KARPARTHY)
      throw new Error("Layers are not supported for non-model modes.")
    if (this.mode === undefined) throw new Error("Model must be initialised before adding layers.")
    if (layers.length === 0) throw new Error("No layers were provided.")

    this.layers = [...this.layers, ...layers]
  }

  /**
   * Compiles the specified model with the given options.
   * @param {Partial<CompileOptions>} options - The compilation options for the model.
   */
  compile(options?: Partial<CompileOptions>): void {
    this.config = { ...DEFAULT_COMPILE_OPTIONS, ...options }

    if (this.mode === undefined)
      throw new Error("Model or table must be initialised before compiling.")

    if (this.mode === Mode.TABLE || this.mode === Mode.KARPARTHY) {
      console.warn('A call to ".compile()" is not required for non-model modes.')
      return
    }

    const { states, actions, layers, config } = this
    this.DQN = new NeuralNetwork(states, actions, layers, config, this.options.alpha)
    this.Target = new NeuralNetwork(states, actions, layers, config, this.options.alpha)
    this.Memory = new Memory(this.options.maxMemory, this.options.batchSize)

    // Copy weights from DQN to Target.
    this.Target.weights(0).set(this.DQN.weights(0).data)

    this.init = true
  }

  /**
   * Trains the model given the input and output data.
   * @param {number[][]} inputs - The inputs to train the model with.
   * @param {number[][]} outputs - The outputs to train the model with.
   */
  train(inputs: number[][], outputs: number[][]): void {
    this.initialised()

    if (inputs.length === 0 || outputs.length === 0 || inputs.length !== outputs.length) {
      throw new Error("The number of inputs and outputs must be equal and greater than 0.")
    }

    // TODO: Train the model with pre-defined data.
  }

  /**
   * Predicts the optimal output given the specified input.
   * @param {number[]} input - The state to predict result for.
   */
  predict(input: number[]): number {
    this.initialised()

    if (this.mode === Mode.MODEL && input.length !== this.states) {
      throw new Error(
        `The number of inputs does not match the required size. (${input.length} !== ${this.states})`
      )
    }

    if (this.mode === Mode.KARPARTHY) return this.KARPATHY.act(input)

    // Greedy epsilon policy.
    let action: number

    if (Math.random() < this.epsilon) action = Math.floor(Math.random() * this.actions)
    else {
      // Determine source from mode.
      if (this.mode === Mode.TABLE) action = Tensor.argmax(this.QTable.get(input))
      else action = Tensor.argmax(this.DQN.forward(input))
    }

    // Decrease epsilon.
    if (this.options.epsilonDecay && this.epsilon > 0) {
      this.epsilon -= this.options.epsilon / this.options.epsilonDecay
    }

    console.log(this.DQN.forward(input))

    // Keep track of this information for the reward phase.
    this.last_state = input
    this.last_action = action
    return action
  }

  /**
   * Rewards the model for the previous action.
   * @param {number} reward - The reward value to give the model.
   * @param {number[]} state - The next state of the model.
   */
  reward(reward: number, state: number[]): void {
    this.initialised()
    this.episode++

    if (this.mode === Mode.KARPARTHY) return this.KARPATHY.learn(reward)

    // Table specific reward logic.
    if (this.mode === Mode.TABLE) {
      const current = this.QTable.get(this.last_state)[this.last_action]
      const next = this.QTable.get(state)

      // Bellman equation.
      const target = reward + this.options.gamma * Math.max(...next)
      const q = current + this.options.alpha * (target - current)
      this.QTable.set(this.last_state, this.last_action, q)
      return

      // Model specific reward logic.
    } else {
      // Add the new sample to memory.
      this.Memory.add(this.last_state, this.last_action, reward, state)

      // Sample a random batch from memory.
      const samples = this.Memory.sample()
      if (!samples) return

      const batch = [...samples, this.Memory.back()]
      for (const b of batch) {
        const target = b.r + this.options.gamma * Math.max(...this.Target.forward(b.n))
        this.DQN.backward(target, b.a, 1 / this.options.batchSize)
      }

      // Copy the weights from the DQN to the Target every episodeSize.
      if (this.episode % this.options.episodeSize === 0) {
        this.Target.weights(0).set(this.DQN.weights(0).data)
      }
    }
  }

  /**
   * Imports model weights from an previously exported source.
   * @param {string} data - The JSON data to import in string form.
   */
  import(data: string): void {
    try {
      const json = JSON.parse(data)

      if (!json.m && !json.w) throw new Error("Invalid data structure for WASML import!")

      if (json.s !== this.states && json.a !== this.actions)
        throw new Error("Imported data does not match specified states and actions!")

      switch (json.m) {
        case Mode.MODEL: {
          this.DQN.load(json.w)
          this.Target.load(json.w)
          break
        }
        case Mode.TABLE: {
          this.QTable.load(json.w)
          break
        }
        case Mode.KARPARTHY: {
          this.KARPATHY.fromJSON(json.w)
          break
        }
        default:
          throw new Error("Invalid mode supplied with data!")
      }
    } catch (e) {
      throw e
    }
  }

  /**
   * Exports the model weights to a JSON string.
   * @returns {string} - The exported model weights and other information.
   */
  export(): string {
    const data = {
      m: this.mode,
      s: this.states,
      a: this.actions,
      w: undefined as any,
    }

    if (this.mode === Mode.MODEL) data.w = this.DQN.save()
    else if (this.mode === Mode.TABLE) data.w = this.QTable.save()
    else if (this.mode === Mode.KARPARTHY) data.w = this.KARPATHY.toJSON()
    return JSON.stringify(data)
  }
}
