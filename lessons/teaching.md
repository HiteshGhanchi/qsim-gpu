# Quantum basics: teaching notes

Personal learning notes for the qsim-gpu project. Written for someone who knows
classical computing (bits, logic gates, building an adder) and is new to quantum.
Pairs with [`lesson0.py`](lesson0.py) (plain Python) and [`lesson1_cuda.py`](lesson1_cuda.py)
(the GPU version): run them while reading.

```bash
python3 lessons/lesson0.py
```

**Contents**
1. [Bits vs qubits: a list of scores](#1-bits-vs-qubits-a-list-of-scores)
2. [Looking at a qubit (measurement)](#2-looking-at-a-qubit-measurement)
3. [Notation you will see everywhere](#3-notation-you-will-see-everywhere)
4. [Gates: small tables that mix scores](#4-gates-small-tables-that-mix-scores)
5. [Interference: the H-H experiment](#5-interference-the-h-h-experiment)
6. [Many qubits: the pair trick](#6-many-qubits-the-pair-trick)
7. [Entanglement: the Bell state](#7-entanglement-the-bell-state)
8. [Complex scores: arrows instead of numbers](#8-complex-scores-arrows-instead-of-numbers)
9. [Gate reference](#9-gate-reference)
10. [Rules of thumb and identities](#10-rules-of-thumb-and-identities)
11. [The three gate families (why the GPU cares)](#11-the-three-gate-families-why-the-gpu-cares)
12. [Glossary](#12-glossary)
13. [Check yourself](#13-check-yourself)
14. [The GPU side: CUDA and CuPy](#14-the-gpu-side-cuda-and-cupy)
15. [Commands you will use](#15-commands-you-will-use)

---

## 1. Bits vs qubits: a list of scores

**Classical:** 3 light switches are always in exactly **one** of 8 patterns
(`000` … `111`). To describe them you only need to say which pattern.

**Quantum:** a qubit register stores a **score for every pattern at once**:

```js
state = { "000": 0.7, "001": 0, "010": 0, ..., "111": 0.7 }
```

In code we drop the keys and use a list, where the **position is the pattern**:

```python
state = [1.0, 0.0]          # 1 qubit:  score of "0", score of "1"
state = [1.0, 0.0, 0.0, 0.0]  # 2 qubits: scores of 00, 01, 10, 11
```

| Qubits | Patterns (list length) | Memory at 8 bytes/score |
|---|---|---|
| 1 | 2 | tiny |
| 10 | 1,024 | 8 KB |
| 20 | ~1 million | 8 MB |
| 30 | ~1 billion | 8 GB |

Each extra qubit **doubles** the list. That doubling is the central problem this
project studies.

A classical bit is just the special case where one score is 1 and the rest are 0:
`[1.0, 0.0]` is a plain bit set to 0.

> In `lesson0.py`: Part 1, `zero: State = [1.0, 0.0]`.

---

## 2. Looking at a qubit (measurement)

When you look at (measure) the qubits:

1. You get **one** pattern, chosen at random.
2. The chance of each pattern is its **score squared** (for arrows, its length squared; section 8).
3. All chances **add up to 1** (100%). Every gate must keep it that way.
4. After looking, the state **collapses** to the pattern you saw. The other scores are gone.

```python
def probabilities(state):
    return [score * score for score in state]   # like state.map(s => s * s)
```

Example: `[0.6, 0.8]` → chances `[0.36, 0.64]`, total 1.

**Collapse matters.** Real hardware can only ever show you one pattern per run, and
looking destroys the rest. A **simulator can read every score without collapsing
anything**. That is one big reason simulators exist: you can debug a circuit by
inspecting the full state.

### Floating-point note
You will see things like `0.6400000000000001` and `0.9999999999999996`. Decimals
cannot be stored exactly in binary (same reason `0.1 + 0.2` is
`0.30000000000000004` in JavaScript). Consequences for the project:

- Never test floats with `==`. Use "close enough": `math.isclose(x, 1, abs_tol=1e-12)`.
- The GPU uses 32-bit floats (`complex64`) to save memory, so its errors are much
  bigger (around the 8th digit instead of the 16th), and they add up over many gates.
- The project **measures** how big that error is on the GPU and sets the test
  tolerance from the measurement, then compares 32-bit and 64-bit in a precision study.

---

## 3. Notation you will see everywhere

| You will see | It means |
|---|---|
| `\|0>` , `\|1>` | "the pattern 0", "the pattern 1" (read: "ket zero") |
| `\|01>` | the 2-qubit pattern `01` |
| amplitude | the score of one pattern |
| statevector | the whole list of scores |
| basis state | one definite pattern, such as `\|01>` (score 1 there, 0 everywhere else) |
| superposition | more than one non-zero score |
| `(\|0> + \|1>)/√2` | score 1/√2 on `0` and 1/√2 on `1`, i.e. `[0.707, 0.707]` |

### Bit order (used everywhere in this project): qubit 0 is the RIGHTMOST bit

| list index | binary | qubit 1 | qubit 0 |
|---|---|---|---|
| 0 | `00` | 0 | 0 |
| 1 | `01` | 0 | **1** |
| 2 | `10` | **1** | 0 |
| 3 | `11` | 1 | 1 |

This is called **little-endian**, the same convention Qiskit uses. It makes
"qubit k" equal to the bit `1 << k`, exactly like a bitmask in JavaScript.

---

## 4. Gates: small tables that mix scores

A **gate** is a small table of numbers (a matrix). A 1-qubit gate is 2×2:

```
new score of "0" = g00 * old"0" + g01 * old"1"
new score of "1" = g10 * old"0" + g11 * old"1"
```

```python
def apply(gate, state):
    a, b = state                          # like: const [a, b] = state
    return [gate[0][0]*a + gate[0][1]*b,
            gate[1][0]*a + gate[1][1]*b]
```

Every row is a **recipe**: "the new score of this pattern is this much of the old
`0` plus this much of the old `1`".

Two rules every gate obeys (the word for this is **unitary**):

1. **Total stays 1.** The chances still add up to 100% after the gate.
2. **Reversible.** Every gate can be undone by another gate. Unlike classical AND,
   nothing is ever thrown away. (`a AND b = 0` doesn't tell you what `a` and `b`
   were; a quantum gate always lets you work backwards.)

> In `lesson0.py`: Part 2, `apply()`. This 4-line function is the heart of the
> whole simulator. The GPU version does exactly this, a billion times in parallel.

---

## 5. Interference: the H-H experiment

The **Hadamard** gate H:

```
H = 1/√2 * [[1,  1],
            [1, -1]]       (1/√2 ≈ 0.707)
```

| Start | After H | After H again |
|---|---|---|
| `[1, 0]` (certainly 0) | `[0.707, 0.707]` (50/50) | `[1, 0]` (certainly 0!) |

After one H the qubit looks like a fair coin. A real coin flipped again stays 50/50.
The qubit instead goes **back to certainly 0**. Do the second H by hand:

- new `0` = 0.707×0.707 + 0.707×0.707 = 0.5 + 0.5 = **1** → the two routes **add**
- new `1` = 0.707×0.707 + 0.707×(−0.707) = 0.5 − 0.5 = **0** → the two routes **cancel**

That is **interference**. Classical probabilities are never negative, so they can
never cancel. Scores can be negative, so they can.

**Every quantum algorithm is a clever arrangement of this cancellation:** make the
routes to wrong answers cancel and the routes to the right answer add up.
Grover and Shor (later phases) are bigger versions of exactly this.

---

## 6. Many qubits: the pair trick

With n qubits the list has 2ⁿ scores. To apply a 1-qubit gate to **qubit k**, you
do **not** need a giant 2ⁿ×2ⁿ table. You walk through **pairs of patterns that
differ only in bit k** and apply the same 2×2 recipe to each pair:

```python
bit = 1 << k                 # qubit 0 -> 0b01, qubit 1 -> 0b10
for i0 in range(len(state)):
    if i0 & bit:             # bit k already 1: it's someone's partner, skip
        continue
    i1 = i0 | bit            # partner: same pattern with bit k switched on
    a, b = state[i0], state[i1]
    state[i0] = g00*a + g01*b
    state[i1] = g10*a + g11*b
```

Example, 2 qubits:

| Gate on | Pairs |
|---|---|
| qubit 0 | (`00`,`01`) and (`10`,`11`) |
| qubit 1 | (`00`,`10`) and (`01`,`11`) |

**Why this is perfect for a GPU:** every pair is independent of every other pair.
No pair needs another's result, so thousands of GPU threads can each take one pair
at the same time, with no waiting and no locks. The CUDA kernel is this exact loop,
written so that thread number `t` computes its own pair directly.

> In `lesson0.py`: Part 3, `apply_on()`.

---

## 7. Entanglement: the Bell state

Recipe: start at `00`, apply **H on qubit 0**, then **CNOT** (control 0, target 1).

```
start      00: 1.000 | 01: 0     | 10: 0 | 11: 0
H on q0    00: 0.707 | 01: 0.707 | 10: 0 | 11: 0
CX q0->q1  00: 0.707 | 01: 0     | 10: 0 | 11: 0.707
```

Readings: only `00` or `11`, never `01` or `10`.

- Qubit 0 alone: a fair coin.
- Qubit 1 alone: a fair coin.
- **But they always agree.**

Two independent fair coins would sometimes give `01`. So this state **cannot** be
described as "qubit 0 is like this" plus "qubit 1 is like that". The only way to
describe it is the **full table**. That is **entanglement**.

**Why it matters for this project:** if qubits could always be described one by one,
n qubits would need only 2n numbers. Because they can be entangled, you need all
2ⁿ. That is why classical simulation is expensive, and why 30 qubits takes 8 GB.

---

## 8. Complex scores: arrows instead of numbers

So far all scores were ordinary (real) numbers: positive or negative. Real quantum
scores are **complex numbers**. You don't need the maths yet; use this picture:

> **A score is an arrow.** It has a **length** and a **direction** (an angle).
> - The chance of a pattern = **length²**. Direction doesn't change the chance.
> - When routes combine, arrows are **added tip to tail**. Arrows pointing the same
>   way add up; arrows pointing opposite ways cancel.

Real numbers are just arrows that can only point right (+) or left (−). Complex
numbers can point in **any** direction, which allows partial cancellation.

- `i` (the "imaginary unit") is an arrow of length 1 pointing **straight up**
  (90°). Multiplying by `i` rotates an arrow by 90°.
- `e^{iφ}` is an arrow of length 1 at angle φ. Multiplying by it **rotates** an arrow by φ.

The direction is called the **phase**.

### Global phase vs relative phase
- **Global phase:** rotate **every** arrow by the same angle. Nothing observable
  changes, ever. (Like rotating the whole drawing.) Two states that differ only by
  a global phase are physically the same.
- **Relative phase:** rotate **some** arrows but not others. Invisible right now
  (lengths unchanged), but it changes how arrows add later, so it changes
  future results. Example: `[0.707, 0.707]` and `[0.707, -0.707]` have identical
  chances, but H turns the first into `0` and the second into `1`.

This is why the project's tests sometimes compare states "up to global phase".

---

## 9. Gate reference

Notation: `|0>` → … means "if the qubit is certainly 0, it becomes …".
All definitions follow Qiskit's conventions, so results can be compared with Qiskit directly.

### 9.1 One-qubit gates

#### X: the NOT gate
```
X = [[0, 1],
     [1, 0]]
```
- `|0>` → `|1>`, `|1>` → `|0>`. Swaps the two scores.
- **Classical analogy:** exactly a NOT gate.
- Its own inverse: X then X = nothing.
- Family: permutation (only moves scores, no maths).

#### H: Hadamard, the "superposition maker"
```
H = 1/√2 * [[1,  1],
            [1, -1]]
```
- `|0>` → `[0.707, 0.707]` (written `|+>`), `|1>` → `[0.707, -0.707]` (written `|->`).
- Both outputs are 50/50; they differ only in the **sign** of the `1` score.
- Its own inverse: H then H = nothing (section 5).
- **No classical analogy.** This is the gate that makes quantum different.
- Used to start almost every algorithm: H on every qubit gives an equal score
  on every pattern ("try all inputs").
- Family: general (mixes scores).

#### Z: phase flip
```
Z = [[1,  0],
     [0, -1]]
```
- Leaves the `0` score alone, flips the sign of the `1` score.
- **Invisible on its own:** chances are unchanged. Visible once H turns signs into
  outcomes: H, Z, H in a row behaves exactly like X.
- Used in Grover to "mark" the answer by flipping its sign.
- Family: diagonal (only multiplies each score by something, never moves it).

#### Y
```
Y = [[0, -i],
     [i,  0]]
```
- `|0>` → `i·|1>`, `|1>` → `−i·|0>`.
- A NOT that also rotates the arrow. Equal to `i · X · Z` (flip the sign of `1`, then swap, then rotate everything by 90°).
- Shows up mostly inside Pauli strings for chemistry (VQE).
- Family: general in this project's grouping (it moves scores *and* multiplies them).

X, Y and Z together are called the **Pauli gates**. A **Pauli string** like
`"XIZY"` means "X on one qubit, nothing (I) on another, Z on another, Y on another".
The chemistry part of the project (VQE) describes a molecule's energy as a
weighted sum of Pauli strings.

#### S, T, P(φ): phase rotations
```
P(φ) = [[1, 0          ],
        [0, e^{iφ}     ]]      rotate the "1" arrow by angle φ

S   = P(90°)    S†  ("Sdg") = P(-90°)
T   = P(45°)    T†  ("Tdg") = P(-45°)
Z   = P(180°)
```
- All of them leave chances unchanged now and only rotate the `1` arrow.
- S twice = Z. T twice = S.
- Sdg undoes S, Tdg undoes T ("dg" = dagger = inverse).
- P(φ) is the general dial; S, T and Z are fixed settings of it.
- Family: diagonal.

#### Rx(θ), Ry(θ), Rz(θ): rotation dials
These have a dial θ (an angle you choose). They are how VQE "tunes" a circuit.
```
Ry(θ) = [[cos θ/2, -sin θ/2],
         [sin θ/2,  cos θ/2]]
```
- **Ry** turns probability smoothly: from `|0>`, the chance of `1` becomes
  sin²(θ/2). θ = 0 does nothing, θ = 90° gives 50/50 (like H on `|0>`),
  θ = 180° gives `|1>`. Real numbers only. Family: general.
- **Rx** does the same with an extra rotation of the arrows (uses `i`).
  Rx(180°) is X up to a global phase. Family: general.
- **Rz(θ)** = diag(e^{−iθ/2}, e^{iθ/2}): rotates the `0` and `1` arrows in opposite
  directions. It equals P(θ) up to a **global** phase, so on its own it behaves
  like P(θ). Careful: once you add a control (section 9.3), that difference stops
  being global and becomes real. Family: diagonal.
- Inverse of any of them: the same gate with −θ.

### 9.2 Two- and three-qubit gates

#### CX (CNOT): controlled NOT
"If the **control** qubit is 1, flip the **target** qubit."

| control | target before | target after |
|---|---|---|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | **1** |
| 1 | 1 | **0** |

- **Classical analogy:** an `if` statement turned into a gate. Also: the target
  becomes `target XOR control`.
- It only **moves** scores (from the pattern to its flipped twin), never changes
  their size. Family: permutation.
- With H before it, it creates entanglement (the Bell state, section 7).
- Its own inverse.

```python
if i & (1 << control):
    new[i] = state[i ^ (1 << target)]   # take the score of the flipped twin
```

#### CZ: controlled Z
"If both qubits are 1, flip the sign of that score." (−1 on `|11>`, nothing else.)
- **Symmetric:** there's no real difference between control and target.
- Same as H on the target, then CX, then H on the target.
- Family: diagonal.

#### CP(φ): controlled phase
"If both qubits are 1, rotate that score by φ." (e^{iφ} on `|11>`.)
- The building block of the **QFT** (quantum Fourier transform), which is the core
  of Shor's algorithm.
- Family: diagonal.

#### SWAP
Exchanges two qubits: `01` ↔ `10`; `00` and `11` stay put.
- **Classical analogy:** `[a, b] = [b, a]`.
- Equal to three CX gates in a row, alternating direction (the XOR-swap trick:
  `a ^= b; b ^= a; a ^= b`).
- Used at the end of the QFT to reverse qubit order.
- Family: permutation.

#### CCX (Toffoli): double-controlled NOT
"If **both** controls are 1, flip the target."
- **Classical analogy:** if the target starts at 0, it ends as `c1 AND c2`. So the
  Toffoli is a **reversible AND gate**. With Toffolis (and NOT) you can build any
  classical circuit, including the adder you already know, reversibly.
- Used in Shor's arithmetic.
- Family: permutation.

### 9.3 Any gate with controls
Any 1-qubit gate can be given one or more controls: "apply this gate to the target
only in patterns where all controls are 1". CX = controlled X, CZ = controlled Z,
CCX = X with two controls. Grover uses a Z with many controls to mark one pattern.

Because controlled gates only act where the controls are 1, they only touch part of
the list (half for one control, a quarter for two). The GPU code can skip the rest.

### 9.4 Summary table

| Gate | Qubits | In one line | Family |
|---|---|---|---|
| X | 1 | NOT: swap the two scores | permutation |
| Y | 1 | NOT plus a rotation | general |
| Z | 1 | flip the sign of the `1` score | diagonal |
| H | 1 | make 50/50 (or undo it); creates interference | general |
| S / Sdg | 1 | rotate the `1` arrow by ±90° | diagonal |
| T / Tdg | 1 | rotate the `1` arrow by ±45° | diagonal |
| P(φ) | 1 | rotate the `1` arrow by φ | diagonal |
| Rx(θ), Ry(θ) | 1 | dial that turns probability between 0 and 1 | general |
| Rz(θ) | 1 | rotate the two arrows opposite ways (≈ P(θ)) | diagonal |
| CX | 2 | if control is 1, flip target | permutation |
| CZ | 2 | sign flip on `11` | diagonal |
| CP(φ) | 2 | rotate the `11` arrow by φ | diagonal |
| SWAP | 2 | exchange two qubits | permutation |
| CCX | 3 | if both controls are 1, flip target (reversible AND) | permutation |

---

## 10. Rules of thumb and identities

Each of these was checked numerically.

| Identity | Meaning |
|---|---|
| H·H = nothing | H undoes itself (the interference demo) |
| X·X = nothing | NOT twice |
| H·Z·H = X | sign flip, sandwiched by H, becomes a NOT |
| H·X·H = Z | and the other way round |
| S·S = Z, T·T = S | phase rotations add up |
| Y = i·X·Z | Y is a NOT plus phases |
| Rz(θ) = (global phase) · P(θ) | same behaviour unless controlled |
| SWAP = CX·CX·CX (alternating) | the XOR-swap trick |
| CZ = H(target)·CX·H(target) | sign flip and NOT are related by H |

**Undoing a whole circuit:** reverse the order and undo each gate
(H→H, X→X, S→Sdg, T→Tdg, Rx(θ)→Rx(−θ)). Running a circuit and then its undo
must return exactly to the start. The project uses this as a test: every circuit followed by its inverse must give back the starting state.

---

## 11. The three gate families (why the GPU cares)

Every gate falls into one of three families, and each family gets its own GPU
code:

| Family | Gates | What happens to the list | Analogy |
|---|---|---|---|
| **Diagonal** | Z, S, T, P, Rz, CZ, CP | each score multiplied by a phase, nothing moves | `arr.map(x => x * phase)` |
| **Permutation** | X, CX, SWAP, CCX | scores move to new positions, no maths | swapping array elements |
| **General** | H, Y, Rx, Ry, any table | pairs of scores mixed with 2×2 maths | the pair trick (section 6) |

**The surprise:** all three cost about the same on a GPU, because each one has to
read and write the **whole** list once. The bottleneck is moving data to and from
memory, not the maths. Like an API that's slow because of the database, not because of
your JavaScript. Measuring exactly that is the project's central question.

---

## 12. Glossary

| Term | Plain meaning |
|---|---|
| Qubit | a quantum bit; a slot in the pattern |
| Amplitude | the score (arrow) of one pattern |
| Statevector | the full list of 2ⁿ scores |
| Basis state | one definite pattern, like `\|01>` |
| Superposition | more than one non-zero score |
| Measurement | looking: get one pattern with chance = length², state collapses |
| Gate | small table that mixes scores; always reversible |
| Unitary | "reversible and keeps the total at 1" |
| Interference | arrows adding up or cancelling |
| Phase | the direction of an arrow |
| Global phase | same rotation on every arrow; never observable |
| Entanglement | qubits that can't be described one at a time |
| Pauli string | a word like `"XIZY"`: one Pauli gate (or nothing) per qubit |
| Little-endian | qubit 0 is the rightmost (lowest) bit of the index |
| Simulator | a program that stores the statevector and applies gates to it |

---

## 13. Check yourself

<details>
<summary>1. Why can classical probabilities never cancel, but quantum scores can?</summary>

Probabilities are never negative, so adding them only ever increases the total.
Scores can be negative (or point in any direction), so two routes can add up to zero.
</details>

<details>
<summary>2. Start from <code>[0.0, 1.0]</code> and apply H. What are the scores and chances?</summary>

`[0.707, -0.707]`, chances 50/50. Same chances as H on `|0>`, but the `1` score is
negative. Apply H again and you get `[0, 1]`: certainly 1. The hidden sign decided
the outcome.
</details>

<details>
<summary>3. In the Bell code, why does <code>apply_cx(after_h, control=1, target=0)</code> do nothing?</summary>

After H on qubit 0, the state has scores only on `00` and `01`. In both, qubit 1
(the left bit) is 0, so the control is never 1 and nothing gets flipped.
</details>

<details>
<summary>4. For H on qubit 1 in a 2-qubit state, which index pairs get mixed?</summary>

Patterns that differ only in bit 1: (`00`,`10`) = indices (0, 2), and
(`01`,`11`) = indices (1, 3).
</details>

<details>
<summary>5. Why does adding one qubit double the memory?</summary>

Each new qubit can be 0 or 1 alongside every existing pattern, so the number of
patterns (and scores) doubles: 2ⁿ → 2ⁿ⁺¹.
</details>

<details>
<summary>6. Z does not change any chance. Why is it useful?</summary>

It flips a sign (a relative phase). Later gates such as H turn that sign into a
different outcome: H·Z·H = X. Grover uses this to mark the answer.
</details>

<details>
<summary>7. Which family is CZ in, and why does that matter for the GPU?</summary>

Diagonal: each score is just multiplied by 1 or −1, nothing moves. The GPU code can
skip the pairing logic entirely, but it still has to read and write the whole list.
</details>

<details>
<summary>8. What is the Toffoli gate in classical terms?</summary>

A reversible AND: with the target starting at 0 it ends as `c1 AND c2`, and the
inputs are kept, so nothing is lost.
</details>


---

## 14. The GPU side: CUDA and CuPy

Run [`lesson1_cuda.py`](lesson1_cuda.py) alongside this section:
```bash
.venv/bin/python lessons/lesson1_cuda.py
```

### Two kinds of memory
Your computer has normal RAM (used by the CPU) and the GPU has its **own** memory. Data must be
copied across, which is slow compared with work done inside either one.

| Code | Where the data lives |
|---|---|
| `np.array(...)` | CPU RAM (NumPy) |
| `cp.asarray(host)` | copies CPU → GPU (CuPy) |
| `device.get()` | copies GPU → CPU |

CuPy arrays look and behave like NumPy arrays, but their numbers live on the GPU and every
operation on them runs there. The simulator keeps the whole state on the GPU and copies back only
when you ask for it (`statevector()`, `probabilities()`), which is why those refuse above 26 qubits.

### What a kernel is
A **kernel** is one small C function that the GPU runs on thousands of threads at once. Each
thread gets its own number and does a tiny piece of the work. The ones in this project are
literally the `apply_on` loop from `lesson0.py`, with the `for` loop removed: instead of one
loop doing every pair, every pair gets its own thread.

```c
long long t   = blockIdx.x * blockDim.x + threadIdx.x;  // "which thread am I?"
long long low = t & ((1LL << k) - 1);
long long i0  = ((t >> k) << (k + 1)) | low;           // my pair's first index
long long i1  = i0 | (1LL << k);                        // ...and its partner
```

- Threads come in **blocks** (here 256 threads each), and blocks form a **grid**.
  `kernel((blocks,), (threads,), args)` launches it.
- `long long` (64-bit) indices: at 30 qubits the numbers get bigger than a 32-bit `int` can hold.
- No two threads touch the same amplitude, so they never wait for each other.

### How the project compiles kernels
The kernels are plain text files (`qsimgpu/kernels/*.cu`). When the simulator starts, CuPy hands
the text to **NVRTC** (NVIDIA's run-time compiler), which turns it into GPU machine code. There's no
separate build step. That's why `nvcc` (the usual CUDA compiler) isn't needed.

### Timing GPU work
GPU calls return immediately; the GPU keeps working in the background. To time it you need GPU
**events** (`cp.cuda.Event()`) and a `synchronize()` to wait for the GPU to finish, as in Part 4
of the lesson. Timing with Python's clock alone would measure only how long it took to *ask*.

### Why bigger is better for a GPU
Launching a kernel has a fixed cost of a few microseconds. With 4 amplitudes that cost is
everything, and NumPy is faster. With 16 million amplitudes, the launch cost disappears and the
GPU's memory speed decides the time. The project measured both (see `docs/DECISIONS.md`): the
4-qubit VQE was slower on the GPU than with NumPy, while big states are where the GPU wins.

---

## 15. Commands you will use

```bash
# One-time setup (Python 3.14 via uv, a project-local environment like node_modules)
uv venv --python 3.14 .venv
uv pip install --python .venv/bin/python -e ".[dev,compare]" "cupy-cuda12x[ctk]"

# Is the GPU visible? Which driver and memory?
nvidia-smi

# Tests
.venv/bin/pytest -m "not gpu"     # CPU tests only
.venv/bin/pytest -m gpu           # GPU tests only
.venv/bin/pytest                  # everything

# Code quality (like eslint + tsc)
.venv/bin/ruff check .
.venv/bin/mypy

# Benchmarks (each writes a JSON file to bench/results/)
.venv/bin/python -m bench.run_env          # record the machine
.venv/bin/python -m bench.run_b2_scaling   # time vs number of qubits
.venv/bin/python -m bench.run_all          # everything, in order

# Lessons
python3 lessons/lesson0.py                 # plain Python, no GPU needed
.venv/bin/python lessons/lesson1_cuda.py   # GPU version
```
