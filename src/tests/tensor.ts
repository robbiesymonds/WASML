import { Tensor } from "../math"

// Some super simple Tensor math tets.
const a = new Tensor([2, 2], [1, 2, 3, 4])
const b = new Tensor([2, 2], [1, 2, 3, 4])
const c = new Tensor([1, 2], [3, 16])
const d = new Tensor([2, 1], [0.3, 0.7])

// Add.
console.log(a.add(b))

// Subtract.
console.log(a.subtract(b))

// Dot product.
console.log(a.dot(2))

// Matrix multiplication.
console.log(c.multiply(d))
