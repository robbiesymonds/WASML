interface Replay {
  c: number[]
  a: number
  r: number
  n: number[]
}

export class Memory {
  private maxSize: number
  private batchSize: number
  private memory: Array<Replay>

  /**
   * Creates a new memory store for experience replay.
   * @constructor
   * @param {number} maxSize - The maximum size of the memory.
   * @param {number} batchSize - The size of the batch to sample for replay.
   */
  constructor(maxSize: number, batchSize: number) {
    this.batchSize = batchSize
    this.maxSize = maxSize
    this.memory = []
  }

  /**
   * Samples the memory for a batch of experiences.
   * @returns {Array<Replay> | null} A random sample of the memory, or null if smaller then batchSize.
   */
  sample(): Array<Replay> | null {
    if (this.memory.length <= this.batchSize) return null

    const set = new Set<number>()

    const { floor, random } = Math
    for (let i = 0; i < this.batchSize; i++) {
      let idx
      do idx = floor(random() * this.memory.length)
      while (set.has(idx))
      set.add(idx)
    }
    return [...set].map((i) => this.memory[i])
  }

  /**
   * Adds a new experience to the memory.
   * @param {number[]} state - The current state of the environment.
   * @param {number} action - The action taken in the environment.
   * @param {number} reward - The reward given for the action.
   * @param {number[]} next - The next state of the environment.
   */
  add(state: number[], action: number, reward: number, next: number[]): void {
    if (this.memory.length + 1 === this.maxSize) this.memory.shift()
    this.memory.push({ c: state, a: action, r: reward, n: next })
  }

  /**
   * Returns the last experience in the memory.
   * @returns {Replay} The latest experience in the memory.
   */
  back(): Replay {
    return this.memory[this.memory.length - 1]
  }

  /**
   * Resets the contents of the memory.
   */
  reset(): void {
    this.memory = new Array(this.maxSize)
  }
}
