import { Tensor } from "../math"

const a = new Tensor([2, 2], [1, 2, 3, 4])
const b = new Tensor([2, 2], [1, 2, 3, 4])

const c = new Tensor([1, 2], [3, 16])
const d = new Tensor([2, 1], [0.3, 0.7])

const add = a.add(b)
console.log(add)

const sub = a.subtract(b)
console.log(sub)

const mul2 = a.dot(2)
console.log(mul2)

const mul = c.multiply(d)
console.log(mul)
