# Lesson 0: a one-qubit "quantum computer" in plain Python.
# Learning file: plain Python, no dependencies, separate from the qsim package.

import math
import random

State = list[float]  # type alias, like `type State = number[]` in TypeScript
Gate = list[list[float]]  # a 2x2 table of numbers


def probabilities(state: State) -> list[float]:
    """Chance of seeing each pattern = its score squared."""
    return [score * score for score in state]


# --- Part 1: your experiment -------------------------------------------
my_state: State = [0.6, 0.8]
print("my_state:     ", my_state, "->", probabilities(my_state))


# --- Part 2: the H gate --------------------------------------------------
S = 1 / math.sqrt(2)  # about 0.7071
H: Gate = [
    [S, S],   # row for new "0": take S of old "0" plus S of old "1"
    [S, -S],  # row for new "1": take S of old "0" MINUS S of old "1"
]


def apply(gate: Gate, state: State) -> State:
    """Each new score is a weighted mix of the old scores."""
    a, b = state  # a = old score of "0", b = old score of "1"
    return [
        gate[0][0] * a + gate[0][1] * b,  # new score of "0"
        gate[1][0] * a + gate[1][1] * b,  # new score of "1"
    ]


zero: State = [1.0, 0.0]  # a plain bit set to 0
once = apply(H, zero)  # H one time
twice = apply(H, once)  # H two times

print("start:        ", zero, "->", probabilities(zero))
print("after H:      ", once, "->", probabilities(once))
print("after H, H:   ", twice, "->", probabilities(twice))


# --- Part 3: two qubits ---------------------------------------------------
def label(index: int, n: int) -> str:
    """Index -> bit pattern, qubit 0 is the RIGHTMOST bit. label(1, 2) == '01'."""
    return format(index, f"0{n}b")


def show(name: str, state: State, n: int) -> None:
    """Print every pattern with its score and its chance."""
    probs = probabilities(state)
    parts = [f"{label(i, n)}: {state[i]:+.3f} ({probs[i]:.0%})" for i in range(len(state))]
    print(f"{name:<14}", " | ".join(parts))


def apply_on(gate: Gate, state: State, k: int) -> State:
    """Apply a 1-qubit gate to qubit k of a bigger state, one pair at a time."""
    new = state.copy()
    bit = 1 << k  # k=0 -> 0b01, k=1 -> 0b10
    for i0 in range(len(state)):
        if i0 & bit:  # bit k already 1: this index is someone's partner, skip
            continue
        i1 = i0 | bit  # partner = same index with bit k switched on
        a, b = state[i0], state[i1]
        new[i0] = gate[0][0] * a + gate[0][1] * b
        new[i1] = gate[1][0] * a + gate[1][1] * b
    return new


def apply_cx(state: State, control: int, target: int) -> State:
    """Controlled NOT: flip qubit `target` wherever qubit `control` is 1."""
    new = state.copy()
    for i in range(len(state)):
        if i & (1 << control):  # control bit is 1 in this pattern
            new[i] = state[i ^ (1 << target)]  # take the score of the flipped pattern
    return new


n = 2
start: State = [1.0, 0.0, 0.0, 0.0]  # 4 scores for patterns 00, 01, 10, 11
after_h = apply_on(H, start, k=0)
bell = apply_cx(after_h, control=0, target=1)

print()
show("start", start, n)
show("H on q0", after_h, n)
show("CX q0->q1", bell, n)

random.seed(1)  # fixed seed so every run prints the same readings
readings = random.choices([label(i, n) for i in range(4)], weights=probabilities(bell), k=10)
print("10 readings:  ", readings)
