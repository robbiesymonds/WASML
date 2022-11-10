const SIZE = 500

export default class FoodGame {
  private axis: number = 0 // 0 = up, 1 = right, 2 = down, 3 = left
  private ctx: CanvasRenderingContext2D
  private player: [number, number]
  private food: [number, number]
  private size: number

  private last_dist!: number

  constructor(size: number) {
    const canvas = document.createElement("canvas")
    this.ctx = canvas.getContext("2d")!
    document.body.appendChild(canvas)
    canvas.height = SIZE
    canvas.width = SIZE
    this.size = size

    this.player = [this.size / 2, this.size / 2]
    this.food = this.random()

    this.render()
  }

  random(): [number, number] {
    return [Math.floor(Math.random() * this.size), Math.floor(Math.random() * this.size)]
  }

  dist() {
    return Math.sqrt((this.food[1] - this.player[1]) ** 2 + (this.food[0] - this.player[0]) ** 2)
  }

  state() {
    const state = {
      axis: this.axis,
      up: this.food[1] < this.player[1] ? 1 : 0,
      down: this.food[1] > this.player[1] ? 1 : 0,
      left: this.food[0] < this.player[0] ? 1 : 0,
      right: this.food[0] > this.player[0] ? 1 : 0,
    }

    this.last_dist = this.dist()
    return Object.values(state)
  }

  move(action: number) {
    switch (action) {
      // Do nothing.
      case 0: {
        if (this.axis === 0) this.player[1] -= 1
        else if (this.axis === 1) this.player[0] += 1
        else if (this.axis === 2) this.player[1] += 1
        else if (this.axis === 3) this.player[0] -= 1
        break
      }
      // Turn left.
      case 1: {
        if (this.axis === 0) this.player[0] -= 1
        else if (this.axis === 1) this.player[1] -= 1
        else if (this.axis === 2) this.player[0] += 1
        else if (this.axis === 3) this.player[1] += 1
        this.axis = (this.axis === 0 ? 3 : this.axis - 1) % 4
        break
      }
      // Turn right.
      case 2: {
        if (this.axis === 0) this.player[0] += 1
        else if (this.axis === 1) this.player[1] += 1
        else if (this.axis === 2) this.player[0] -= 1
        else if (this.axis === 3) this.player[1] -= 1
        this.axis = (this.axis + 1) % 4
      }
    }
  }

  reward() {
    const S = this.size

    // Check if collided with wall and respawn if so.
    if (this.player[0] < 0 || this.player[0] >= S || this.player[1] < 0 || this.player[1] >= S) {
      this.player = [S / 2, S / 2]
      return -10.0
    }

    //  Check has got the food and respawn if so.
    if (this.player[0] === this.food[0] && this.player[1] === this.food[1]) {
      this.food = this.random()
      return 100.0
    }

    // Otherwise negative reward.
    return this.dist() < this.last_dist ? 0.05 : -0.1
  }

  render() {
    const U = SIZE / this.size
    this.ctx.fillStyle = "#000"
    this.ctx.fillRect(0, 0, SIZE, SIZE)

    this.ctx.fillStyle = "#FFF"
    this.ctx.fillRect(this.player[0] * U, this.player[1] * U, U, U)

    this.ctx.fillStyle = "#F00"
    this.ctx.fillRect(this.food[0] * U, this.food[1] * U, U, U)
  }

  loop(callback: (state: number[]) => void) {
    requestAnimationFrame(() => {
      callback(this.state())
      this.loop(callback)
    })
  }
}
