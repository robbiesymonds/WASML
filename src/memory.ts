interface Replay {
  c: Float32Array
  a: number
  r: number
  n: Float32Array
}

export class Memory {
  private maxSize: number
  private batchSize: number
  private memory: Array<Replay>

  /**
   * Creates a new memory store for experience replay.
   * @constructor
   * @param {number} maxSize The maximum size of the memory.
   * @param {number} batchSize The size of the batch to sample for replay.
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
   * @param {Float32Array} state The current state of the environment.
   * @param {number} action The action taken in the environment.
   * @param {number} reward The reward given for the action.
   * @param {Float32Array} next The next state of the environment.
   */
  add(state: Float32Array, action: number, reward: number, next: Float32Array): void {
    if (this.memory.length + 1 === this.maxSize) this.memory.shift()
    this.memory.push({ c: state, a: action, r: reward, n: next })
  }

  /**
   * Resets the contents of the memory.
   */
  reset(): void {
    this.memory = new Array(this.maxSize)
  }

  /**
   * Loads a pre-existing model brain into the network.
   * @param {Array<Replay> data The memory object to load.
   */
  load(data: Array<Replay>): void {
    this.memory = data
  }

  /**
   * Exports the current memory store.
   * @returns The current state of the memory object.
   */
  save(): Array<Replay> {
    return this.memory
  }
}
