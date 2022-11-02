import { CompileOptions } from "./index"
import {
  derivative,
  evaluate,
  map,
  matrix,
  Matrix,
  multiply,
  random,
  subtract,
  sum,
  transpose,
} from "mathjs"

export interface Layer {
  units: number
  activation: "sigmoid" | "relu" | "linear" | "softmax"
}

export interface LayerInternal extends Layer {
  weights: number[]
}

export class NeuralNetwork {
  private layers: LayerInternal[]
  private states: number
  private actions: number
  private options: CompileOptions
  private alpha: number

  // Allows state to preserve the last state and action.
  private cache: { input: number[]; output: number[]; weights: number[]; gradients: number[] } = {
    input: [],
    output: [],
    weights: [],
    gradients: [],
  }

  /**
   * Build the neural network with the specified options.
   * @constructor
   * @param {number} states The number of input states.
   * @param {number} actions The number of output actions.
   * @param {Layer[]} layers The hidden layers to give the neural network.
   */
  constructor(
    states: number,
    actions: number,
    layers: Layer[],
    options: CompileOptions,
    alpha: number
  ) {
    this.states = states
    this.actions = actions
    this.options = options
    this.alpha = alpha
    this.layers = layers.map((l) => ({
      ...l,
      weights: Array.from({ length: this.states }, () => random(0, 0.5)),
    }))
  }

  /**
   * Returns the weights for the specified layer.
   * @param {number} idx The index of the layer to get the weights for.
   * @returns {number[]} The weights for the specified layer.
   */
  weights(idx: number): number[] {
    return this.layers[idx].weights
  }

  /**
   * Performs a forward pass activation through the network.
   * @param {number[]} input The input state through the network.
   * @param {Layer["activation"]} type The type of activation to use. (sigmoid, relu, ...)
   * @returns {number[]} Result of the activation.
   */
  private activate(input: number[], type: Layer["activation"]): number[] {
    switch (type) {
      case "linear":
        return input
      case "softmax":
        return input.map((x) => Math.exp(x) / sum(input.map((x) => Math.exp(x))))
      case "relu":
        return input.map((x) => Math.max(0, x))
      case "sigmoid":
        return input.map((x) => 1 / (1 + Math.exp(-x)))
    }
  }

  /**
   * Calculates the Q-values for the given input.
   * @param {number[]} input The current state to give the network.
   * @returns {Matrix} The optimal Q-value for the given input.
   */
  forward(input: number[]): number[] {
    const z = evaluate("w .* x", { w: this.weights(0), x: input })
    const output = this.activate(z, this.layers[0].activation)

    // Save values to cache.
    this.cache = { ...this.cache, input, output, weights: this.weights(0) }

    return output
  }

  /**
   * Back-propagates the loss and updates weights.
   * @param {number[]} target The target state of the network.
   */
  backward(target: number[]): void {
    // Calculate gradient of loss with respect to activation.
    const deda = subtract(target, this.cache.output)
    console.log(deda)

    // Calculate the gradient of the activation with respect to the weights.
    const dadz = this.cache.output.map((x) => (x > 0 ? 1 : 0))
    console.log(dadz)
    // const dadz: Matrix = evaluate("1", {
    //   x: matrix([this.cache.output]),
    // })

    // Calculate the gradient of the loss with respect to the weights.
    const dedw: Matrix = evaluate("(deda .* dadz) * w", { deda, dadz, w: this.cache.weights })
    console.log(dedw)

    this.cache.gradients = dedw.toArray() as number[]
  }

  descend(): void {
    this.layers[0].weights = evaluate("w + (a .* g)", {
      w: this.weights(0),
      a: this.alpha,
      g: this.cache.gradients,
    })

    console.log(this.weights(0))
  }

  fit(target: number[]): void {
    this.backward(target)
    this.descend()
  }
}
