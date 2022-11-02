import init, { setup } from "../dist/wasml"
import { Memory } from "./memory"
import { Layer, NeuralNetwork } from "./network"

interface ModelOptions {
  alpha: number
  gamma: number
  epsilon: number
  maxMemory: number
  batchSize: number
  episodeSize: number
}

export interface CompileOptions {
  loss: "meanSquaredError" | "hinge"
  optimizer: "sgd" | "adam"
}

// The default configurations for the model.
const DEFAULT_MODEL_OPTIONS: ModelOptions = {
  alpha: 0.1,
  gamma: 0.6,
  epsilon: 0.1,
  maxMemory: 1000,
  batchSize: 100,
  episodeSize: 50,
}

const DEFAULT_COMPILE_OPTIONS: CompileOptions = {
  loss: "meanSquaredError",
  optimizer: "sgd",
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

  private last_state!: Float32Array
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

  /*
   * Check that the model has been initialized.
   */
  initialised(): void {
    if (!this.init) {
      throw new Error("A call to `.compile()` is required before interaction is performed!")
    }
  }

  /**
   * Adds the specified layers to the neural network.
   * @param {Layer[]} layers - The array of layers to add to the neural network.
   */
  addLayers(layers: Layer[]): void {
    this.initialised()
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
    this.Target.weights(0).set(this.DQN.weights(0))

    this.init = true
  }

  /**
   * Trains the model given the input and output data.
   * @param {Float32Array[]} inputs - The inputs to train the model with.
   * @param {Float32Array[]} outputs - The outputs to train the model with.
   */
  train(inputs: Float32Array[], outputs: Float32Array[]): void {
    this.initialised()

    if (inputs.length === 0 || outputs.length === 0 || inputs.length !== outputs.length) {
      throw new Error("The number of inputs and outputs must be equal and greater than 0.")
    }

    console.log(inputs, outputs)
  }

  /**
   * Predicts the optimal output given the specified input.
   * @param {Float32Array} input - The state to predict result for.
   */
  predict(input: Float32Array): number {
    this.initialised()

    if (input.length !== this.states) {
      throw new Error(
        `The number of inputs does not match the required size. (${input.length} !== ${this.states})`
      )
    }

    // Greedy epsilon policy.
    let action: number
    if (Math.random() < this.options.epsilon) action = Math.floor(Math.random() * this.actions)
    else action = this.DQN.action(input)

    // Keep track of this information for the reward phase.
    this.last_state = input
    this.last_action = action

    return action
  }

  /**
   * Rewards the model for the previous action.
   * @param {Float32Array} state - The state to reward the model for.
   * @param {number} reward - The reward value to give the model.
   */
  reward(state: Float32Array, reward: number): void {
    this.initialised()
    this.episode++

    // Add the new sample to memory.
    this.Memory.add(this.last_state, this.last_action, reward, state)

    // Sample a random batch from memory.
    const batch = this.Memory.sample()
    if (!batch) return

    console.log(batch)
    const predicted_qs = batch.map((b) => this.DQN.q(b.c))
    const target_qs = batch.map((b) => this.Target.q(b.n))

    batch.forEach((b) => {
      const predicted_q = predicted_qs[b.a]
      const target_q = b.r + this.options.gamma * Math.max(...target_qs)
      const loss = (predicted_q - target_q) ** 2
      this.DQN.fit(loss)
    })

    // Copy the weights from the DQN to the Target every episodeSize.
    if (this.episode % this.options.episodeSize === 0) {
      this.Target.weights(0).set(this.DQN.weights(0))
    }
  }
}
