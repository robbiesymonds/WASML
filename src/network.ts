import { CompileOptions } from "./index"
import { Activation, Loss, Tensor } from "./math"

export interface Layer {
  units: number
  activation: "sigmoid" | "relu" | "linear" | "tanh"
}

export interface LayerInternal extends Layer {
  weights: Tensor
}

export class NeuralNetwork {
  private layers: LayerInternal[]
  private options: CompileOptions
  private actions: number
  private states: number
  private alpha: number

  private cache!: Tensor

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

    if (layers.length === 0) throw new Error("At least one hidden layer is required.")

    if (layers[layers.length - 1].units !== actions) {
      throw new Error("The last layer units must match the number of actions.")
    }

    this.layers = layers.map((l, i) => {
      const input = i === 0 ? states : layers[i - 1].units
      return {
        ...l,
        weights: new Tensor(
          [l.units, input],
          Array.from({ length: input * l.units }, () => Math.random() * 0.5)
        ),
      }
    })
  }

  /**
   * Returns the weights of specified layer in the network.
   * @param {number} idx The index of the layer to get the weights of.
   * @returns The weights of the specified layer.
   */
  weights(idx: number): Tensor {
    return this.layers[idx].weights
  }

  /**
   * Performs the activation function for the given input.
   * @param {Tensor} input The input tensor to perform the activation function on.
   * @param {Layer["activation"]} type The type of activation to use. (sigmoid, relu, ...)
   * @returns {Tensor} Result of the activation.
   */
  private activate(input: Tensor, type: Layer["activation"]): Tensor {
    return Activation[type].fn(input)
  }

  /**
   * Calculates the forward pass out of specified state.
   * @param {number[]} input The current state to give the network.
   * @returns {Tensor} The result for the given input.
   */
  forward(input: number[]): number[] {
    const x = new Tensor([this.states, 1], input)
    let z: Tensor = x

    this.layers.forEach((l) => {
      z = l.weights.multiply(z)
      z = this.activate(z, l.activation)
    })

    this.cache = x
    return z.data
  }

  /**
   * Back-propagates the loss and updates weights.
   * @param {number[]} target The target state of the network.
   * @param {number} action - Optional index to only reward specific action.
   * @param {number} batchSize - Optional batch size to use for scaling impact of loss.
   */
  backward(target: number[], action?: number, batchSize?: number): void {
    const input = this.cache
    const N = this.layers.length

    const outputs: Tensor[] = [input]
    const errors: Tensor[] = []

    // Forward pass to get outputs of each layer.
    this.layers.forEach((l, i) => {
      let z = l.weights.multiply(outputs[i])
      z = this.activate(z, l.activation)
      outputs.push(z)
    })

    // Derivative of the loss function.
    target = target.map((t, i) => (action ? (i === action ? outputs[N].data[i] : t) : t))
    errors[N] = Loss[this.options.loss].derivative(
      outputs[N],
      new Tensor([this.actions, 1], target)
    )

    if (batchSize) errors[N] = errors[N].dot(1 / batchSize)

    // From last layer, propagate backwards.
    for (let i = N; i > 0; i--) {
      let gradient = Activation[this.layers[i - 1].activation].derivative(outputs[i])
      gradient = gradient.dot(this.alpha)
      gradient = gradient.dot(errors[i])

      // Update weights.
      const delta = gradient.multiply(outputs[i - 1].transpose())
      this.layers[i - 1].weights = this.layers[i - 1].weights.add(delta)

      // Calculate error for next layer.
      errors[i - 1] = this.layers[i - 1].weights.transpose().multiply(errors[i])
    }
  }
}
