import init, { setup } from "../dist/wasml"
import { Memory } from "./memory"
import { Layer, NeuralNetwork } from "./network"
import { Tensor } from "./math"

interface ModelOptions {
  alpha: number
  gamma: number
  epsilon: number
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
  gamma: 0.9,
  epsilon: 0.1,
  maxMemory: 1000,
  batchSize: 100,
  episodeSize: 50,
}

const DEFAULT_COMPILE_OPTIONS: CompileOptions = {
  loss: "meanSquaredError",
}

export default class WASML {
  private init: boolean = false
  private states: number = Infinity
  private actions: number = Infinity
  private options: ModelOptions = DEFAULT_MODEL_OPTIONS
  private config: CompileOptions = DEFAULT_COMPILE_OPTIONS
  private layers: Layer[] = []

  private DQN!: NeuralNetwork
  private Target!: NeuralNetwork
  private Memory!: Memory

  private last_state!: number[]
  private last_action!: number
  private episode: number = 0

  /**
   * Generates a model with the given options.
   * @param {number} states - The size of the input vector space for the model.
   * @param {number} actions - The number of actions the model can take.
   * @param {ModelOptions} options - The options for the model.
   */
  async model(states: number, actions: number, options?: Partial<ModelOptions>): Promise<void> {
    this.states = states
    this.actions = actions

    await init().then(() => {
      const config = { ...DEFAULT_MODEL_OPTIONS, ...options }
      setup(actions, states)
      this.options = config
    })
  }

  /**
   * Checks if the model has been initialised.
   * @returns {boolean} - Whether the model has been initialised.
   */
  private initialised(): void {
    if (!this.init) {
      throw new Error("A call to `.compile()` is required before interaction is performed!")
    }
  }

  /**
   * Adds the specified layers to the neural network.
   * @param {Layer[]} layers - The array of layers to add to the neural network.
   */
  addLayers(layers: Layer[]): void {
    if (layers.length === 0) throw new Error("No layers were provided.")
    this.layers = [...this.layers, ...layers]
  }

  /**
   * Compiles the specified model with the given options.
   * @param {Partial<CompileOptions>} options - The compilation options for the model.
   */
  compile(options: Partial<CompileOptions>): void {
    this.config = { ...DEFAULT_COMPILE_OPTIONS, ...options }

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

    if (input.length !== this.states) {
      throw new Error(
        `The number of inputs does not match the required size. (${input.length} !== ${this.states})`
      )
    }

    // Greedy epsilon policy.
    let action: number

    if (Math.random() < this.options.epsilon) action = Math.floor(Math.random() * this.actions)
    else action = Tensor.argmax(this.DQN.forward(input))

    console.log(this.DQN.forward(input))

    // Keep track of this information for the reward phase.
    this.last_state = input
    this.last_action = action
    return action
  }

  /**
   * Rewards the model for the previous action.
   * @param {number[]} state - The next state of the model.
   * @param {number} reward - The reward value to give the model.
   */
  reward(reward: number, state: number[]): void {
    this.initialised()
    this.episode++

    // Add the new sample to memory.
    this.Memory.add(this.last_state, this.last_action, reward, state)

    // Sample a random batch from memory.
    const batch = this.Memory.sample()
    if (!batch) return

    for (const b of batch) {
      const target = new Tensor([this.actions, 1], this.Target.forward(b.n))
        .dot(this.options.gamma)
        .add(reward).data

      this.DQN.backward(target, b.a, this.options.batchSize)
    }

    // Copy the weights from the DQN to the Target every episodeSize.
    if (this.episode % this.options.episodeSize === 0) {
      this.Target.weights(0).set(this.DQN.weights(0).data)
    }
  }
}
