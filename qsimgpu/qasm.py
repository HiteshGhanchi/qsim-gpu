"""OpenQASM 2.0 import and export, written by hand (no dependencies).

    from qsimgpu.qasm import from_qasm2, to_qasm2
    circuit = from_qasm2(open("bell.qasm").read())
    text = to_qasm2(circuit)

Supported on import:
- `OPENQASM 2.0;`, `include "qelib1.inc";` (the standard library is built in; no other includes)
- several `qreg` / `creg` declarations (flattened in declaration order into one register each)
- the qelib1 gates that map to qsimgpu, plus `U` and `CX`: u3 u2 u1 u cx id x y z h s sdg t tdg
  rx ry rz p sx sxdg cz cy ch swap ccx cswap crx cry crz cu1 cp cu3 csx rxx rzz, and the extra
  gates of the qelib1.inc that Qiskit writes against: u0 cu c3x c3sqrtx c4x rccx rc3x
- `gate` definitions (expanded like macros) with parameter expressions: numbers, pi, + - * / ^,
  unary minus, parentheses and sin cos tan exp ln sqrt
- `measure a -> b;`, `reset a;`, `barrier ...;`, and `if (creg == value) <operation>;`
- whole-register arguments, broadcast as the specification describes

Mapping choices: u3/u/U -> u, u2(phi, lam) -> u(pi/2, phi, lam), u1 -> p, cu1 -> cp, and the
controlled forms (cy, ch, crx, ...) -> the base gate with a control. rzz(t) is replaced by its
exact decomposition cx, rz(t), cx; rxx(t) by h h, rzz(t), h h. `id` and `barrier` do nothing in a
simulator and are dropped on purpose; anything else that is not understood is an error with its
line and column, never skipped. `opaque` gates cannot be simulated and are an error too.

Export writes qelib1 names (u3, u1, cu1, cu3, ...). It raises QasmExportError for what OpenQASM 2
cannot express: `unitary` gates, more than one control (except ccx), and conditions that don't
match a classical register.

OpenQASM 3 is not handled here; see qsimgpu.interop.qiskit (optional, needs qiskit).
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from qsimgpu.circuit import Circuit, Gate


class QasmError(ValueError):
    """Syntax or semantic error in OpenQASM input, with its position."""

    def __init__(self, message: str, line: int, col: int) -> None:
        super().__init__(f"line {line}, column {col}: {message}")
        self.line, self.col = line, col


class QasmExportError(ValueError):
    """The circuit uses something OpenQASM 2 cannot express."""


# --- tokens ------------------------------------------------------------------------------

@dataclass(frozen=True)
class Tok:
    kind: str  # "id", "num", "str", "sym", "eof"
    text: str
    line: int
    col: int


_TOKEN_RE = re.compile(r"""
    (?P<ws>[ \t\r]+) | (?P<nl>\n) | (?P<comment>//[^\n]*)
  | (?P<num>(\d+\.\d*|\.\d+|\d+)([eE][-+]?\d+)?)
  | (?P<id>[A-Za-z_][A-Za-z0-9_]*)
  | (?P<str>"[^"\n]*")
  | (?P<sym>->|==|[;,()\[\]{}+\-*/^])
""", re.VERBOSE)


def tokenize(text: str) -> list[Tok]:
    toks: list[Tok] = []
    line, line_start, pos = 1, 0, 0
    while pos < len(text):
        m = _TOKEN_RE.match(text, pos)
        col = pos - line_start + 1
        if not m:
            raise QasmError(f"unexpected character {text[pos]!r}", line, col)
        kind = m.lastgroup
        assert kind is not None
        if kind == "nl":
            line, line_start = line + 1, m.end()
        elif kind not in ("ws", "comment"):
            toks.append(Tok(kind, m.group(), line, col))
        pos = m.end()
    toks.append(Tok("eof", "", line, pos - line_start + 1))
    return toks


# --- expressions ---------------------------------------------------------------------------

Expr = Callable[[dict[str, float]], float]
_FUNCS: dict[str, Callable[[float], float]] = {
    "sin": math.sin, "cos": math.cos, "tan": math.tan, "exp": math.exp, "ln": math.log,
    "sqrt": math.sqrt,
}


# --- gate definitions ------------------------------------------------------------------------

@dataclass
class Call:
    name: str
    params: list[Expr]
    args: list[str]  # names of the definition's qubit arguments
    tok: Tok


@dataclass
class GateDef:
    params: list[str]
    args: list[str]
    body: list[Call] = field(default_factory=list)


_BUILTIN_ARITY: dict[str, tuple[int, int]] = {  # name -> (number of params, number of qubits)
    "U": (3, 1), "CX": (0, 2), "u3": (3, 1), "u": (3, 1), "u2": (2, 1), "u1": (1, 1),
    "cx": (0, 2), "id": (0, 1), "x": (0, 1), "y": (0, 1), "z": (0, 1), "h": (0, 1), "s": (0, 1),
    "sdg": (0, 1), "t": (0, 1), "tdg": (0, 1), "rx": (1, 1), "ry": (1, 1), "rz": (1, 1),
    "p": (1, 1), "sx": (0, 1), "sxdg": (0, 1), "cz": (0, 2), "cy": (0, 2), "ch": (0, 2),
    "swap": (0, 2), "ccx": (0, 3), "cswap": (0, 3), "crx": (1, 2), "cry": (1, 2), "crz": (1, 2),
    "cu1": (1, 2), "cp": (1, 2), "cu3": (3, 2), "csx": (0, 2), "rxx": (1, 2), "rzz": (1, 2),
    # In the qelib1.inc that Qiskit writes against (it declares them without defining them):
    "u0": (1, 1), "cu": (4, 2), "c3x": (0, 4), "c3sqrtx": (0, 4), "c4x": (0, 5),
}

# Qiskit's qelib1.inc also provides these relative-phase Toffolis; their standard definitions:
_LIBRARY = """
gate rccx a,b,c { u2(0,pi) c; u1(pi/4) c; cx b,c; u1(-pi/4) c; cx a,c; u1(pi/4) c; cx b,c;
                  u1(-pi/4) c; u2(0,pi) c; }
gate rc3x a,b,c,d { u2(0,pi) d; u1(pi/4) d; cx c,d; u1(-pi/4) d; u2(0,pi) d; cx a,d;
                    u1(pi/4) d; cx b,d; u1(-pi/4) d; cx a,d; u1(pi/4) d; cx b,d; u1(-pi/4) d;
                    u2(0,pi) d; u1(pi/4) d; cx c,d; u1(-pi/4) d; u2(0,pi) d; }
"""


def _builtin(name: str, p: list[float], q: list[int]) -> list[Gate]:
    """qsimgpu gates for one qelib1 gate application (parameters already evaluated)."""
    one = {"x", "y", "z", "h", "s", "sdg", "t", "tdg", "rx", "ry", "rz", "p", "sx", "sxdg"}
    if name in one:
        return [Gate(name, (q[0],), params=tuple(p))]
    if name in ("U", "u3", "u"):
        return [Gate("u", (q[0],), params=tuple(p))]
    if name == "u2":
        return [Gate("u", (q[0],), params=(math.pi / 2, p[0], p[1]))]
    if name == "u1":
        return [Gate("p", (q[0],), params=(p[0],))]
    if name in ("id", "u0"):  # u0(n) is an idle of n time steps: the identity here
        return []
    if name == "cu":  # controlled U(theta, phi, lambda) with an extra phase gamma on the control
        return [Gate("p", (q[0],), params=(p[3],)), Gate("u", (q[1],), (q[0],), tuple(p[:3]))]
    if name in ("c3x", "c4x"):
        return [Gate("x", (q[-1],), tuple(q[:-1]))]
    if name == "c3sqrtx":
        return [Gate("sx", (q[-1],), tuple(q[:-1]))]
    if name in ("CX", "cx"):
        return [Gate("cx", (q[1],), (q[0],))]
    if name == "cz":
        return [Gate("cz", (q[1],), (q[0],))]
    if name in ("cu1", "cp"):
        return [Gate("cp", (q[1],), (q[0],), (p[0],))]
    controlled = {"cy": "y", "ch": "h", "crx": "rx", "cry": "ry", "crz": "rz", "csx": "sx",
                  "cu3": "u"}
    if name in controlled:
        return [Gate(controlled[name], (q[1],), (q[0],), tuple(p))]
    if name == "swap":
        return [Gate("swap", (q[0], q[1]))]
    if name == "ccx":
        return [Gate("ccx", (q[2],), (q[0], q[1]))]
    if name == "cswap":
        return [Gate("swap", (q[1], q[2]), (q[0],))]
    if name == "rzz":
        return [Gate("cx", (q[1],), (q[0],)), Gate("rz", (q[1],), params=(p[0],)),
                Gate("cx", (q[1],), (q[0],))]
    if name == "rxx":
        h = [Gate("h", (q[0],)), Gate("h", (q[1],))]
        return h + _builtin("rzz", p, q) + h
    raise AssertionError(name)  # every name in _BUILTIN_ARITY is handled above


# --- parser ------------------------------------------------------------------------------------

class _Parser:
    def __init__(self, text: str, library: bool = True) -> None:
        self.defs: dict[str, GateDef] = {}
        if library:
            lib = _Parser("OPENQASM 2.0;\n" + _LIBRARY, library=False)
            lib.take("OPENQASM")
            lib.take(kind="num")
            lib.take(";")
            while lib.tok.kind != "eof":
                lib.take("gate")
                lib.gate_definition()
            self.defs = lib.defs
        self.toks = tokenize(text)
        self.i = 0
        self.qregs: dict[str, tuple[int, int]] = {}  # name -> (offset, size)
        self.cregs: dict[str, tuple[int, int]] = {}
        self.n_qubits = self.n_clbits = 0
        self.ops: list[Gate] = []

    # token helpers
    @property
    def tok(self) -> Tok:
        return self.toks[self.i]

    def error(self, msg: str, tok: Tok | None = None) -> QasmError:
        t = tok or self.tok
        return QasmError(msg, t.line, t.col)

    def take(self, text: str | None = None, kind: str | None = None) -> Tok:
        t = self.tok
        if (text is not None and t.text != text) or (kind is not None and t.kind != kind):
            want = repr(text) if text is not None else kind
            raise self.error(f"expected {want}, found {t.text or 'end of file'!r}")
        self.i += 1
        return t

    def accept(self, text: str) -> bool:
        if self.tok.text == text and self.tok.kind != "str":
            self.i += 1
            return True
        return False

    # expressions: sum := term (('+'|'-') term)*; term := unary (('*'|'/') unary)*;
    # unary := ('-'|'+') unary | power; power := atom ('^' unary)?
    # so -pi^2 = -(pi^2) and 2^-1 = 0.5, as in ordinary maths
    def expr(self, names: Sequence[str]) -> Expr:
        left = self.term(names)
        while self.tok.text in ("+", "-"):
            op = self.take().text
            right = self.term(names)
            left = (lambda a, b: lambda e: a(e) + b(e))(left, right) if op == "+" else (
                lambda a, b: lambda e: a(e) - b(e))(left, right)
        return left

    def term(self, names: Sequence[str]) -> Expr:
        left = self.unary(names)
        while self.tok.text in ("*", "/"):
            op_tok = self.take()
            right = self.unary(names)
            if op_tok.text == "*":
                left = (lambda a, b: lambda e: a(e) * b(e))(left, right)
            else:
                left = self._divide(left, right, op_tok)
        return left

    def _divide(self, a: Expr, b: Expr, tok: Tok) -> Expr:
        def f(e: dict[str, float]) -> float:
            d = b(e)
            if d == 0:
                raise QasmError("division by zero", tok.line, tok.col)
            return a(e) / d
        return f

    def power(self, names: Sequence[str]) -> Expr:
        base = self.atom(names)
        if self.accept("^"):
            exp = self.unary(names)
            return lambda e: base(e) ** exp(e)
        return base

    def unary(self, names: Sequence[str]) -> Expr:
        if self.accept("-"):
            inner = self.unary(names)
            return lambda e: -inner(e)
        if self.accept("+"):
            return self.unary(names)
        return self.power(names)

    def atom(self, names: Sequence[str]) -> Expr:
        t = self.tok
        if t.kind == "num":
            self.i += 1
            value = float(t.text)
            return lambda e: value
        if t.kind == "id":
            self.i += 1
            if t.text == "pi":
                return lambda e: math.pi
            if t.text in _FUNCS:
                self.take("(")
                arg = self.expr(names)
                self.take(")")
                fn = _FUNCS[t.text]
                return lambda e: fn(arg(e))
            if t.text in names:
                name = t.text
                return lambda e: e[name]
            raise self.error(f"unknown identifier {t.text!r} in expression", t)
        if self.accept("("):
            inner = self.expr(names)
            self.take(")")
            return inner
        raise self.error(f"expected an expression, found {t.text or 'end of file'!r}")

    def expr_list(self, names: Sequence[str]) -> list[Expr]:
        out: list[Expr] = []
        if self.accept("("):
            if not self.accept(")"):
                out.append(self.expr(names))
                while self.accept(","):
                    out.append(self.expr(names))
                self.take(")")
        return out

    # registers and arguments
    def declare(self, kind: str) -> None:
        name = self.take(kind="id")
        self.take("[")
        size_tok = self.take(kind="num")
        self.take("]")
        self.take(";")
        if not size_tok.text.isdigit() or int(size_tok.text) < 1:
            raise self.error("register size must be a positive integer", size_tok)
        if name.text in self.qregs or name.text in self.cregs:
            raise self.error(f"register {name.text!r} declared twice", name)
        size = int(size_tok.text)
        if kind == "qreg":
            self.qregs[name.text] = (self.n_qubits, size)
            self.n_qubits += size
        else:
            self.cregs[name.text] = (self.n_clbits, size)
            self.n_clbits += size

    def argument(self, regs: dict[str, tuple[int, int]], what: str) -> list[int]:
        """A register argument: list of absolute indices (one element for name[i])."""
        name = self.take(kind="id")
        if name.text not in regs:
            raise self.error(f"unknown {what} register {name.text!r}", name)
        offset, size = regs[name.text]
        if self.accept("["):
            idx = self.take(kind="num")
            self.take("]")
            if not idx.text.isdigit() or int(idx.text) >= size:
                raise self.error(f"index {idx.text} out of range for {name.text}[{size}]", idx)
            return [offset + int(idx.text)]
        return list(range(offset, offset + size))

    @staticmethod
    def broadcast(args: list[list[int]], tok: Tok) -> list[list[int]]:
        sizes = {len(a) for a in args if len(a) > 1}
        if len(sizes) > 1:
            raise QasmError("registers of different sizes in one statement", tok.line, tok.col)
        n = sizes.pop() if sizes else 1
        return [[a[k] if len(a) > 1 else a[0] for a in args] for k in range(n)]

    # statements
    def parse(self) -> Circuit:
        self.take("OPENQASM")
        version = self.take(kind="num")
        if not version.text.startswith("2"):
            raise self.error(f"only OpenQASM 2 is supported here, got {version.text}", version)
        self.take(";")
        while self.tok.kind != "eof":
            self.statement()
        if self.n_qubits == 0:
            raise self.error("no qreg declared")
        c = Circuit(self.n_qubits, self.ops, n_clbits=self.n_clbits)
        c.cregs = {k: tuple(range(o, o + s)) for k, (o, s) in self.cregs.items()}
        return c

    def statement(self) -> None:
        t = self.tok
        if self.accept("include"):
            f = self.take(kind="str")
            self.take(";")
            if f.text.strip('"') != "qelib1.inc":
                raise self.error(f"only qelib1.inc can be included, not {f.text}", f)
        elif t.text in ("qreg", "creg"):
            self.i += 1
            self.declare(t.text)
        elif self.accept("gate"):
            self.gate_definition()
        elif t.text == "opaque":
            raise self.error("opaque gates have no definition and cannot be simulated")
        elif self.accept("barrier"):
            self.argument_list(self.qregs, "quantum")
            self.take(";")
        elif self.accept("if"):
            self.take("(")
            reg = self.take(kind="id")
            if reg.text not in self.cregs:
                raise self.error(f"unknown classical register {reg.text!r}", reg)
            self.take("==")
            val = self.take(kind="num")
            self.take(")")
            offset, size = self.cregs[reg.text]
            if not val.text.isdigit() or int(val.text) >= (1 << size):
                raise self.error(f"value {val.text} does not fit in {reg.text}[{size}]", val)
            cond = (tuple(range(offset, offset + size)), int(val.text))
            self.quantum_op(cond)
        else:
            self.quantum_op(None)

    def argument_list(self, regs: dict[str, tuple[int, int]], what: str) -> list[list[int]]:
        args = [self.argument(regs, what)]
        while self.accept(","):
            args.append(self.argument(regs, what))
        return args

    def emit(self, gates: list[Gate], cond: tuple[tuple[int, ...], int] | None, tok: Tok) -> None:
        for g in gates:
            try:
                self.ops.append(g.with_condition(*cond) if cond else g)
            except ValueError as exc:
                raise self.error(str(exc), tok) from None

    def quantum_op(self, cond: tuple[tuple[int, ...], int] | None) -> None:
        t = self.tok
        if self.accept("measure"):
            q = self.argument(self.qregs, "quantum")
            self.take("->")
            c = self.argument(self.cregs, "classical")
            self.take(";")
            if len(q) != len(c):
                raise self.error("measure: registers of different sizes", t)
            if cond and len(q) > 1 and set(c) & set(cond[0]):
                # The condition is tested once for the whole statement, but qsimgpu tests it per
                # measurement; they differ only if the measurements write the tested register.
                raise self.error("conditional broadcast measure into its own condition register "
                                 "is not supported", t)
            self.emit([Gate("measure", (a,), clbits=(b,)) for a, b in zip(q, c, strict=True)],
                      cond, t)
        elif self.accept("reset"):
            q = self.argument(self.qregs, "quantum")
            self.take(";")
            self.emit([Gate("reset", (a,)) for a in q], cond, t)
        elif t.kind == "id":
            self.i += 1
            params = [p({}) for p in self.expr_list([])]
            args = self.argument_list(self.qregs, "quantum")
            self.take(";")
            for qubits in self.broadcast(args, t):
                if len(set(qubits)) != len(qubits):
                    raise self.error(f"{t.text}: the same qubit is used twice", t)
                self.emit(self.expand(t.text, params, qubits, t), cond, t)
        else:
            raise self.error(f"unexpected {t.text or 'end of file'!r}")

    def gate_definition(self) -> None:
        name = self.take(kind="id")
        params: list[str] = []
        if self.accept("("):
            if not self.accept(")"):
                params.append(self.take(kind="id").text)
                while self.accept(","):
                    params.append(self.take(kind="id").text)
                self.take(")")
        args = [self.take(kind="id").text]
        while self.accept(","):
            args.append(self.take(kind="id").text)
        d = GateDef(params, args)
        self.take("{")
        while not self.accept("}"):
            t = self.tok
            if self.accept("barrier"):
                while self.tok.text != ";":
                    self.i += 1
                self.take(";")
                continue
            callee = self.take(kind="id")
            exprs = self.expr_list(params)
            call_args = [self.take(kind="id")]
            while self.accept(","):
                call_args.append(self.take(kind="id"))
            self.take(";")
            for a in call_args:
                if a.text not in args:
                    raise self.error(f"unknown qubit argument {a.text!r} in gate {name.text}", a)
            if callee.text not in _BUILTIN_ARITY and callee.text not in self.defs:
                raise self.error(f"gate {callee.text!r} is not defined", callee)
            d.body.append(Call(callee.text, exprs, [a.text for a in call_args], t))
        self.defs[name.text] = d

    def expand(self, name: str, params: list[float], qubits: list[int], tok: Tok) -> list[Gate]:
        if name in self.defs:
            d = self.defs[name]
            if (len(params), len(qubits)) != (len(d.params), len(d.args)):
                raise self.error(f"{name} takes {len(d.params)} parameter(s) and "
                                 f"{len(d.args)} qubit(s)", tok)
            env = dict(zip(d.params, params, strict=True))
            amap = dict(zip(d.args, qubits, strict=True))
            out: list[Gate] = []
            for call in d.body:
                out += self.expand(call.name, [e(env) for e in call.params],
                                   [amap[a] for a in call.args], call.tok)
            return out
        if name not in _BUILTIN_ARITY:
            raise self.error(f"gate {name!r} is not supported and not defined", tok)
        n_params, n_qubits = _BUILTIN_ARITY[name]
        if (len(params), len(qubits)) != (n_params, n_qubits):
            raise self.error(f"{name} takes {n_params} parameter(s) and {n_qubits} qubit(s)", tok)
        return _builtin(name, params, qubits)


def from_qasm2(text: str) -> Circuit:
    """Parse an OpenQASM 2.0 program into a Circuit (raises QasmError with line and column)."""
    return _Parser(text).parse()


# --- export ---------------------------------------------------------------------------------------

_EXPORT_ONE = {"h": "h", "x": "x", "y": "y", "z": "z", "s": "s", "sdg": "sdg", "t": "t",
               "tdg": "tdg", "rx": "rx", "ry": "ry", "rz": "rz", "p": "u1", "sx": "sx",
               "sxdg": "sxdg", "u": "u3"}
_EXPORT_CONTROLLED = {"x": "cx", "z": "cz", "y": "cy", "h": "ch", "rx": "crx", "ry": "cry",
                      "rz": "crz", "p": "cu1", "u": "cu3", "sx": "csx"}


def _fmt(x: float) -> str:
    return repr(float(x))


def to_qasm2(circuit: Circuit) -> str:
    """Write a Circuit as OpenQASM 2.0 using qelib1 gates (raises QasmExportError)."""
    lines = ["OPENQASM 2.0;", 'include "qelib1.inc";', f"qreg q[{circuit.n_qubits}];"]
    cregs = dict(getattr(circuit, "cregs", {}) or {})
    if not cregs and circuit.n_clbits:
        conds = {g.condition[0] for g in circuit if g.condition}
        if all(len(bits) == 1 for bits in conds):
            cregs = {f"c{j}": (j,) for j in range(circuit.n_clbits)}
        else:
            cregs = {"c": tuple(range(circuit.n_clbits))}
    where = {b: (name, k) for name, bits in cregs.items() for k, b in enumerate(bits)}
    lines += [f"creg {name}[{len(bits)}];" for name, bits in cregs.items()]
    by_bits = {bits: name for name, bits in cregs.items()}
    for g in circuit:
        prefix = ""
        if g.condition:
            bits, value = g.condition
            if bits not in by_bits:
                raise QasmExportError(f"condition on clbits {bits} is not a whole classical "
                                      "register; OpenQASM 2 can only test whole registers")
            prefix = f"if({by_bits[bits]}=={value}) "
        lines.append(prefix + _export_op(g, where))
    return "\n".join(lines) + "\n"


def _export_op(g: Gate, where: dict[int, tuple[str, int]]) -> str:
    q = [f"q[{i}]" for i in g.targets]
    c = [f"q[{i}]" for i in g.controls]
    params = f"({','.join(_fmt(p) for p in g.params)})" if g.params else ""
    if g.name == "measure":
        name, k = where[g.clbits[0]]
        return f"measure {q[0]} -> {name}[{k}];"
    if g.name == "reset":
        return f"reset {q[0]};"
    if g.name == "unitary":
        raise QasmExportError("'unitary' gates (arbitrary matrices) have no OpenQASM 2 form")
    if g.name == "swap":
        if not g.controls:
            return f"swap {q[0]},{q[1]};"
        if len(g.controls) == 1:
            return f"cswap {c[0]},{q[0]},{q[1]};"
    base = {"cx": "x", "cz": "z", "cp": "p", "ccx": "x"}.get(g.name, g.name)
    if not g.controls and base in _EXPORT_ONE:
        return f"{_EXPORT_ONE[base]}{params} {q[0]};"
    if len(g.controls) == 1 and base in _EXPORT_CONTROLLED:
        return f"{_EXPORT_CONTROLLED[base]}{params} {c[0]},{q[0]};"
    if len(g.controls) == 2 and base == "x":
        return f"ccx {c[0]},{c[1]},{q[0]};"
    raise QasmExportError(f"{g.name} with {len(g.controls)} control(s) has no OpenQASM 2 "
                          "qelib1 form")
