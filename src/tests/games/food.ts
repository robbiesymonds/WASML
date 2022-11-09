const SIZE = 500
const GRID = 30

export default class FoodGame {
  private axis: number = 0 // 0 = up, 1 = right, 2 = down, 3 = left
  private ctx: CanvasRenderingContext2D
  private player: [number, number] = [GRID / 2, GRID / 2]
  private food: [number, number] = [GRID / 2, GRID / 2]

  constructor() {
    const canvas = document.createElement("canvas")
    this.ctx = canvas.getContext("2d")!
    document.body.appendChild(canvas)
    canvas.width = SIZE
    canvas.height = SIZE

    this.render()
  }

  state() {
    const state = {
      axis: this.axis,
      up: this.food[1] < this.player[1] ? 1 : 0,
      down: this.food[1] > this.player[1] ? 1 : 0,
      left: this.food[0] < this.player[0] ? 1 : 0,
      right: this.food[0] > this.player[0] ? 1 : 0,
      dist: Math.floor(
        Math.sqrt((this.food[1] - this.player[1]) ** 2 + (this.food[0] - this.player[0]) ** 2)
      ),
    }

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
    // Check if collided with wall.
    if (
      this.player[0] < 0 ||
      this.player[0] >= GRID ||
      this.player[1] < 0 ||
      this.player[1] >= GRID
    ) {
      // Respawn the player.
      this.player = [GRID / 2, GRID / 2]
      return -10.0
    }

    //  Check has got the food.
    if (this.player[0] === this.food[0] && this.player[1] === this.food[1]) {
      // Respawn the food elsewhere.
      this.food = [Math.floor(Math.random() * GRID), Math.floor(Math.random() * GRID)]
      return 100.0
    }

    // Otherwise negative reward, the closer the food, the less penalty.
    return (
      (-1 *
        Math.floor(
          Math.sqrt((this.food[1] - this.player[1]) ** 2 + (this.food[0] - this.player[0]) ** 2)
        )) /
      GRID
    )
  }

  render() {
    const U = SIZE / GRID
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
