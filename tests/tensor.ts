import { Tensor } from "../src/math"
import WASML from "../src/index"

const main = async () => {
  // Must be run to initialise WASM memory.
  await WASML.load()

  // Some super simple Tensor math tests.
  const a = new Tensor([2, 2], [1, 2, 3, 4])
  const b = new Tensor([1, 2], [3, 16])
  const c = new Tensor([2, 1], [0.3, 0.7])

  // Add.
  console.log(a.add(a))

  // Add scalar.
  console.log(a.add(5))

  // Subtract.
  console.log(a.subtract(a))

  // Subtract scalar.
  console.log(a.subtract(0.5))

  // Transpose.
  console.log(a.transpose())

  // Argmax.
  console.log(Tensor.argmax(a.data))

  // Sum.
  console.log(b.sum())

  // Dot product.
  console.log(b.dot(b))

  // Dot product scalar.
  console.log(a.dot(4))

  // Matrix multiplication.
  console.log(b.multiply(c))

  // Matrix multiplication.
  console.log(a.multiply(a))
}

main()
