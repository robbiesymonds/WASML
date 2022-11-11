export class Table {
  private table: Record<string, number[]> = {}
  private actions: number
  private states: number

  /**
   * Create a new dictionary-based table abstraction.
   * @param {number} states - Number of rows.
   * @param {number} actions - Number of columns.
   * @constructor
   */
  constructor(states: number, actions: number) {
    this.actions = actions
    this.states = states
  }

  /**
   * Get the column values for a given state (row).
   * @param {number[]} state - The state to lookup.
   * @returns {number[]} - The column values for the given state.
   */
  get(state: number[]): number[] {
    if (state.length !== this.states) {
      throw new Error(`State must be ${this.states} units but got ${state.length}!`)
    }

    const key = state.join(",")
    let q: number[]

    if (this.table[key]) {
      return this.table[key]
    } else return new Array(this.actions).fill(0)
  }

  /**
   * Sets the column value for a given state (row) and action (column).
   * @param {number[]} state - The state to update.
   * @param {number} action - The action (index) to update.
   * @param {number} value - The new value to insert.
   */
  set(state: number[], action: number, value: number): void {
    if (state.length !== this.states) {
      throw new Error(`State must be ${this.states} units but got ${state.length}!`)
    }

    const key = state.join(",")
    if (this.table[key]) {
      this.table[key][action] = value
    } else {
      this.table[key] = new Array(this.actions).fill(0)
      this.table[key][action] = value
    }
  }

  /**
   * Loads a pre-existing Q-Table.
   * @param {Record<string, number[]>} table - The table to load.
   */
  load(table: Record<string, number[]>): void {
    this.table = table
  }

  /**
   * Returns the current state of the Q-Table.
   * @returns {Record<string, number[]>} - The table.
   */
  save(): Record<string, number[]> {
    return this.table
  }
}
