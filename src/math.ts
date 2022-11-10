import { CompileOptions } from "."
import { Layer } from "./network"
import {
  add_matrix,
  add_scalar,
  subtract_matrix,
  subtract_scalar,
  argmax,
  dot_matrix,
  dot_scalar,
  transpose,
  sum,
  multiply,
} from "../dist/wasml"

export class Tensor {
  data: number[]
  shape: [number, number]

  /**
   * Creates a new tensor with the specified shape.
   * @param {[number, number]} shape - The rows and columns of the tensor.
   * @param {number[]} data - The data for the tensor.
   */
  constructor(shape: [number, number], data: number[]) {
    if (shape[0] * shape[1] !== data.length)
      throw new Error("Data must match the shape of the tensor.")

    this.shape = shape
    this.data = data
  }

  /**
   * Explicitly sets the data for the tensor.
   * @param {number[]} data - The data to use for the tensor.
   */
  set(data: number[]): void {
    if (data.length !== this.shape[0] * this.shape[1]) {
      throw new Error("Data must match the shape of the tensor.")
    }

    this.data = data
  }

  /**
   * Retuns the index of the maximum value in an array.
   * @param {number[]} arr - The array to find the maximum value in.
   * @returns {number} - The index of the maximum value in the tensor.
   */
  static argmax(arr: number[]): number {
    return argmax(new Float64Array(arr)) ?? 0
  }

  /**
   * Gets the data value at the specified row and column.
   * @param i - The row to get.
   * @param j - The column to get.
   * @returns {number} - The value at the specified row and column.
   */
  private at(i: number, j: number): number {
    return this.data[i * this.shape[1] + j]
  }

  /**
   * Checks that the current tensor shape is equal to the specified tensor.
   * @param {Tensor} t - The tensor to check the shape against.
   * @returns {boolean} - Whether or not the matrices are equal.
   */
  private compare(t: Tensor): boolean {
    if (this.shape[0] !== t.shape[0] || this.shape[1] !== t.shape[1]) {
      throw new Error("Dimensions of tensors must match.")
    }
    return true
  }

  /**
   * Adds the specified tensor or number to the current tensor.
   * @param {Tensor | number} other - The tensor or number to add to the current tensor.
   * @returns {Tensor} - The resulting tensor.
   */
  add(other: Tensor | number): Tensor {
    let data = new Float64Array(this.data)
    if (other instanceof Tensor) this.compare(other)
    if (other instanceof Tensor) data = add_matrix(data, new Float64Array(other.data))
    else data = add_scalar(data, other)

    return new Tensor(this.shape, Array.from(data))
  }

  /**
   * Subtracts the specified tensor to the current tensor.
   * @param {Tensor} other - The tensor to subtract from the current tensor.
   * @returns {Tensor} - The resulting tensor.
   */
  subtract(other: Tensor | number): Tensor {
    let data = new Float64Array(this.data)
    if (other instanceof Tensor) this.compare(other)
    if (other instanceof Tensor) data = subtract_matrix(data, new Float64Array(other.data))
    else data = subtract_scalar(data, other)

    return new Tensor(this.shape, Array.from(data))
  }

  /**
   * Multiplies the current tensor by the specified tensor.
   * @param {Tensor} other - The tensor to multiply the current tensor by.
   * @returns {Tensor} - The resulting tensor.
   */
  multiply(other: Tensor): Tensor {
    if (this.shape[1] !== other.shape[0])
      throw new Error("Invalid dimensions for matrix multiplication.")

    let data = multiply(
      new Float64Array(this.data),
      new Float64Array(other.data),
      this.shape[0],
      other.shape[1],
      this.shape[1]
    )

    return new Tensor([this.shape[0], other.shape[1]], Array.from(data))
  }

  /**
   * Calculates the element-wise multiplication of the current tensor and the specified tensor.
   * @param {Tensor | number} other - The tensor or number to multiply the current tensor by.
   * @returns {Tensor} - The resulting tensor.
   */
  dot(other: Tensor | number): Tensor {
    let data = new Float64Array(this.data)
    if (other instanceof Tensor) this.compare(other)
    if (other instanceof Tensor) data = dot_matrix(data, new Float64Array(other.data))
    else data = dot_scalar(data, other)

    return new Tensor(this.shape, Array.from(data))
  }

  /**
   * Transposes the current tensor.
   * @returns {Tensor} - The transposed tensor.
   */
  transpose(): Tensor {
    const data = transpose(new Float64Array(this.data), this.shape[0], this.shape[1])
    return new Tensor([this.shape[1], this.shape[0]], Array.from(data))
  }

  /**
   * Calculates the sum of the current tensor.
   * @returns {number} - The sum of all the values in the tensor.
   */
  sum(): number {
    return sum(new Float64Array(this.data))
  }

  /**
   * Maps the current tensor to the specified function.
   * @param {Function} fn - The function to map the tensor to.
   * @returns {Tensor} - The resulting tensor.
   */
  map(fn: (x: number, i: number) => number): Tensor {
    const data = this.data.map(fn)
    return new Tensor(this.shape, data)
  }
}

export const Activation: Record<
  Layer["activation"],
  { fn: (x: Tensor) => Tensor; derivative: (x: Tensor) => Tensor }
> = {
  linear: {
    fn: (x) => x,
    derivative: (x) => x.map(() => 1),
  },
  relu: {
    fn: (x) => x.map((n) => Math.max(0, n)),
    derivative: (x) => x.map((n) => (n > 0 ? 1 : 0)),
  },
  sigmoid: {
    fn: (x) => x.map((n) => 1 / (1 + Math.exp(-n))),
    derivative: (x) => x.map((n) => (1 / (1 + Math.exp(-n))) * (1 - 1 / (1 + Math.exp(-n)))),
  },
  tanh: {
    fn: (x) => x.map((n) => Math.tanh(n)),
    derivative: (x) => x.map((n) => 1 - Math.tanh(n) ** 2),
  },
}

export const Loss: Record<
  CompileOptions["loss"],
  { fn: (x: Tensor, y: Tensor) => number; derivative: (x: Tensor, y: Tensor) => Tensor }
> = {
  meanSquaredError: {
    fn: (x, y) =>
      y
        .subtract(x)
        .map((n) => n * n)
        .sum() / x.data.length,
    derivative: (x, y) => y.subtract(x),
  },
  meanAbsoluteError: {
    fn: (x, y) =>
      y
        .subtract(x)
        .map((n) => Math.abs(n))
        .sum() / x.data.length,
    derivative: (x, y) => y.subtract(x).map((n) => (n > 0 ? 1 : -1)),
  },
}
