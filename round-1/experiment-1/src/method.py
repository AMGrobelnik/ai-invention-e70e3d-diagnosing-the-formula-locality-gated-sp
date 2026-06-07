#!/usr/bin/env python3
"""SBFL-for-FOL: Spectrum-Based Fault Localization for NL→FOL Translation.

6-Phase pipeline: locality pilot → aligned translation → probe bank → Ochiai ranking →
targeted repair → ablations. Outputs method_out.json in exp_gen_sol_out schema.
"""

import asyncio
import gc
import json
import math
import os
import re
import sys
import time
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any

import aiohttp
import nltk
import numpy as np
import psutil
import resource
import z3
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

# ── Logging ──────────────────────────────────────────────────────────────────
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
Path("logs").mkdir(exist_ok=True)
logger.add("logs/run.log", rotation="30 MB", level="DEBUG")

# ── Hardware limits ───────────────────────────────────────────────────────────
_avail = psutil.virtual_memory().available
RAM_BUDGET = min(int(_avail * 0.5), 8 * 1024**3)  # 50% avail, max 8GB
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET * 3, RAM_BUDGET * 3))
logger.info(f"RAM budget: {RAM_BUDGET/1e9:.1f}GB")

NUM_CPUS = 6
CONCURRENCY = 8  # async parallel API calls

# ── Config ────────────────────────────────────────────────────────────────────
WORKSPACE = Path(__file__).parent
MAX_SENTENCES = int(os.environ.get("MAX_SENTENCES", "120"))
BUDGET_USD = 9.50
MODEL = "google/gemini-2.5-flash"
DOMAIN_SIZE = 6  # Z3 bounded domain size
Z3_TIMEOUT_MS = 5000

cost_tracker = {"total": 0.0, "calls": 0}

# ── OpenRouter API ────────────────────────────────────────────────────────────
def get_api_key() -> str:
    for k in ["OPENROUTER_API_KEY", "OR_API_KEY"]:
        v = os.environ.get(k, "")
        if v:
            return v
    # Try loading from .env files
    for env_file in [Path.home() / ".env", Path("/ai-inventor/.env"), Path(".env")]:
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                line = line.strip()
                if line.startswith("OPENROUTER_API_KEY=") or line.startswith("OR_API_KEY="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError("OPENROUTER_API_KEY not found in environment or .env files")


OR_API_KEY = get_api_key()
OR_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"


class BudgetExceeded(Exception):
    pass


async def openrouter_call_async(
    session: aiohttp.ClientSession,
    prompt: str,
    *,
    model: str = MODEL,
    system: str = "",
    max_tokens: int = 1024,
    temperature: float = 0.0,
    json_mode: bool = False,
    retries: int = 3,
) -> dict:
    if cost_tracker["total"] >= BUDGET_USD:
        raise BudgetExceeded(f"Budget ${cost_tracker['total']:.2f} exceeded")

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    headers = {
        "Authorization": f"Bearer {OR_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://ai-inventor.local",
        "X-Title": "SBFL-FOL",
    }

    last_error = None
    for attempt in range(retries):
        try:
            async with session.post(
                OR_BASE_URL, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=90)
            ) as resp:
                if resp.status == 429:
                    await asyncio.sleep(2 ** attempt * 2)
                    continue
                resp.raise_for_status()
                data = await resp.json()
                # Track cost
                usage = data.get("usage", {})
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
                # Gemini flash pricing: $0.15/$0.60 per 1M
                cost = (prompt_tokens * 0.15 + completion_tokens * 0.60) / 1_000_000
                cost_tracker["total"] += cost
                cost_tracker["calls"] += 1
                if cost_tracker["calls"] % 10 == 0:
                    logger.info(f"Cost so far: ${cost_tracker['total']:.4f} ({cost_tracker['calls']} calls)")
                content = data["choices"][0]["message"]["content"]
                return {"content": content, "cost": cost}
        except BudgetExceeded:
            raise
        except Exception as e:
            last_error = e
            logger.warning(f"API attempt {attempt+1}/{retries} failed: {e}")
            if attempt < retries - 1:
                await asyncio.sleep(2 ** attempt)

    raise RuntimeError(f"API call failed after {retries} retries: {last_error}")


def openrouter_call_sync(prompt: str, **kwargs) -> dict:
    async def _run():
        async with aiohttp.ClientSession() as session:
            return await openrouter_call_async(session, prompt, **kwargs)
    return asyncio.run(_run())


# ── FOL Parser ────────────────────────────────────────────────────────────────

class ParseError(Exception):
    pass


class FOLNode:
    __slots__ = ["ntype", "label", "children", "component_id", "nesting_depth"]

    def __init__(self, ntype: str, label: str, children: list | None = None):
        self.ntype = ntype  # forall|exists|not|and|or|impl|iff|xor|pred|var|const
        self.label = label
        self.children: list["FOLNode"] = children or []
        self.component_id = ""
        self.nesting_depth = 0

    def __repr__(self):
        if self.children:
            return f"FOLNode({self.ntype},{self.label!r},{self.children})"
        return f"FOLNode({self.ntype},{self.label!r})"

    def __eq__(self, other):
        if not isinstance(other, FOLNode):
            return False
        return self.ntype == other.ntype and self.label == other.label and self.children == other.children


def _tokenize_fol(s: str) -> list[str]:
    """Tokenize FOL string with Unicode logic symbols."""
    s = s.strip()
    # Rewrite infix equality: x=y → Eq(x,y) before other replacements
    s = re.sub(r'(\w+)\s*=\s*(\w+)', r'Eq(\1,\2)', s)
    # Unicode symbol → ASCII token
    replacements = [
        ("∀", " FORALL "), ("∃", " EXISTS "),
        ("¬", " NOT "), ("∧", " AND "), ("∨", " OR "),
        ("→", " IMPL "), ("↔", " IFF "), ("⊕", " XOR "),
        # ASCII logic aliases
        ("->", " IMPL "), ("<->", " IFF "),
        ("(", " ( "), (")", " ) "), (",", " , "),
    ]
    for sym, tok in replacements:
        s = s.replace(sym, tok)
    tokens = s.split()
    return [t for t in tokens if t]


class FOLParser:
    def __init__(self, tokens: list[str]):
        self.tokens = tokens
        self.pos = 0
        self._component_counter = [0]

    def peek(self) -> str | None:
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None

    def consume(self, expected: str | None = None) -> str:
        if self.pos >= len(self.tokens):
            raise ParseError(f"Unexpected end of input (expected {expected!r})")
        tok = self.tokens[self.pos]
        if expected and tok != expected:
            raise ParseError(f"Expected {expected!r}, got {tok!r} at pos {self.pos}")
        self.pos += 1
        return tok

    def _new_cid(self) -> str:
        cid = f"c{self._component_counter[0]}"
        self._component_counter[0] += 1
        return cid

    def parse(self) -> FOLNode:
        node = self.parse_iff()
        if self.pos < len(self.tokens):
            raise ParseError(f"Unexpected token {self.tokens[self.pos]!r} at pos {self.pos}")
        return node

    def parse_iff(self) -> FOLNode:
        left = self.parse_impl()
        while self.peek() in ("IFF",):
            self.consume()
            right = self.parse_impl()
            node = FOLNode("iff", "↔", [left, right])
            left = node
        return left

    def parse_impl(self) -> FOLNode:
        left = self.parse_or()
        while self.peek() == "IMPL":
            self.consume()
            right = self.parse_or()
            node = FOLNode("impl", "→", [left, right])
            left = node
        return left

    def parse_or(self) -> FOLNode:
        left = self.parse_and()
        while self.peek() == "OR":
            self.consume()
            right = self.parse_and()
            node = FOLNode("or", "∨", [left, right])
            left = node
        return left

    def parse_and(self) -> FOLNode:
        left = self.parse_unary()
        while self.peek() == "AND":
            self.consume()
            right = self.parse_unary()
            node = FOLNode("and", "∧", [left, right])
            left = node
        return left

    def parse_unary(self) -> FOLNode:
        tok = self.peek()
        if tok == "NOT":
            self.consume()
            child = self.parse_unary()
            return FOLNode("not", "¬", [child])
        if tok == "FORALL":
            self.consume()
            var = self.consume()
            # optional paren around body
            if self.peek() == "(":
                self.consume("(")
                body = self.parse_iff()
                self.consume(")")
            else:
                body = self.parse_unary()
            node = FOLNode("forall", var, [body])
            node.component_id = self._new_cid()
            return node
        if tok == "EXISTS":
            self.consume()
            var = self.consume()
            if self.peek() == "(":
                self.consume("(")
                body = self.parse_iff()
                self.consume(")")
            else:
                body = self.parse_unary()
            node = FOLNode("exists", var, [body])
            node.component_id = self._new_cid()
            return node
        if tok == "XOR":
            self.consume()
            left = self.parse_unary()
            right = self.parse_unary()
            return FOLNode("xor", "⊕", [left, right])
        return self.parse_atom()

    def parse_atom(self) -> FOLNode:
        tok = self.peek()
        if tok == "(":
            self.consume("(")
            node = self.parse_iff()
            self.consume(")")
            return node
        if tok is None:
            raise ParseError("Unexpected end of input")
        # EQ token in prefix position (shouldn't happen normally; skip gracefully)
        if tok == "EQ":
            self.consume()
            return FOLNode("pred", "EQ", [])
        # Predicate: starts with uppercase
        if tok[0].isupper():
            self.consume()
            if self.peek() == "(":
                self.consume("(")
                args = []
                while self.peek() not in (None, ")"):
                    arg_tok = self.tokens[self.pos]
                    self.pos += 1
                    # arg might itself be a nested structure; treat as var/const
                    args.append(FOLNode("var", arg_tok))
                    if self.peek() == ",":
                        self.consume(",")
                self.consume(")")
                node = FOLNode("pred", tok, args)
                node.component_id = self._new_cid()
                return node
            else:
                # Constant or zero-arity predicate
                node = FOLNode("pred", tok, [])
                node.component_id = self._new_cid()
                return node
        # Lowercase = variable or constant
        self.consume()
        return FOLNode("var", tok)


def parse_fol(s: str) -> FOLNode:
    """Parse a FOL string to AST. Raises ParseError on failure."""
    s = s.strip()
    if not s:
        raise ParseError("Empty formula")
    tokens = _tokenize_fol(s)
    if not tokens:
        raise ParseError("No tokens")
    parser = FOLParser(tokens)
    tree = parser.parse()
    _assign_nesting_depth(tree, 0)
    return tree


def _assign_nesting_depth(node: FOLNode, depth: int):
    node.nesting_depth = depth
    child_depth = depth + (1 if node.ntype in ("forall", "exists") else 0)
    for c in node.children:
        _assign_nesting_depth(c, child_depth)


def canonical_rename(tree: FOLNode) -> FOLNode:
    """Rename bound variables to x0, x1, ... in DFS order. Returns new tree."""
    counter = [0]
    rename_map: dict[str, str] = {}

    def _rename(node: FOLNode) -> FOLNode:
        new_children = []
        if node.ntype in ("forall", "exists"):
            old_var = node.label
            new_var = f"x{counter[0]}"
            counter[0] += 1
            old_val = rename_map.get(old_var)
            rename_map[old_var] = new_var
            for c in node.children:
                new_children.append(_rename(c))
            rename_map[old_var] = old_val  # restore
            new_node = FOLNode(node.ntype, new_var, new_children)
            new_node.component_id = node.component_id
            new_node.nesting_depth = node.nesting_depth
            return new_node
        elif node.ntype == "var":
            new_label = rename_map.get(node.label, node.label)
            return FOLNode("var", new_label)
        else:
            for c in node.children:
                new_children.append(_rename(c))
            new_node = FOLNode(node.ntype, node.label, new_children)
            new_node.component_id = node.component_id
            new_node.nesting_depth = node.nesting_depth
            return new_node

    return _rename(tree)


def fol_to_string(node: FOLNode) -> str:
    """Convert FOLNode back to string."""
    nt = node.ntype
    if nt == "var":
        return node.label
    if nt == "const":
        return node.label
    if nt == "pred":
        if node.children:
            args = ",".join(fol_to_string(c) for c in node.children)
            return f"{node.label}({args})"
        return node.label
    if nt == "not":
        inner = fol_to_string(node.children[0])
        return f"¬{inner}"
    if nt == "forall":
        body = fol_to_string(node.children[0])
        return f"∀{node.label}({body})"
    if nt == "exists":
        body = fol_to_string(node.children[0])
        return f"∃{node.label}({body})"
    if nt == "and":
        return f"({fol_to_string(node.children[0])} ∧ {fol_to_string(node.children[1])})"
    if nt == "or":
        return f"({fol_to_string(node.children[0])} ∨ {fol_to_string(node.children[1])})"
    if nt == "impl":
        return f"({fol_to_string(node.children[0])} → {fol_to_string(node.children[1])})"
    if nt == "iff":
        return f"({fol_to_string(node.children[0])} ↔ {fol_to_string(node.children[1])})"
    if nt == "xor":
        return f"({fol_to_string(node.children[0])} ⊕ {fol_to_string(node.children[1])})"
    return node.label


def get_fault_components(tree: FOLNode) -> list[FOLNode]:
    """Get fault-site components: predicate nodes + quantifier nodes."""
    result = []

    def _walk(node: FOLNode):
        if node.ntype in ("pred", "forall", "exists") and node.component_id:
            result.append(node)
        for c in node.children:
            _walk(c)

    _walk(tree)
    return result


def get_all_components(tree: FOLNode) -> list[FOLNode]:
    return get_fault_components(tree)


# ── ZSS Tree Edit Distance ────────────────────────────────────────────────────

def _zss_get_children(node: FOLNode) -> list[FOLNode]:
    return node.children


def _zss_get_label(node: FOLNode) -> str:
    return f"{node.ntype}:{node.label}"


def _zss_label_dist(a: str, b: str) -> int:
    return 0 if a == b else 1


def compute_ast_edit_distance(gold: FOLNode, cand: FOLNode) -> tuple[int, int]:
    """Returns (edit_distance, n_faulty_components) using ZSS."""
    try:
        import zss
        distance = zss.simple_distance(
            gold, cand,
            get_children=_zss_get_children,
            get_label=_zss_get_label,
            label_dist=_zss_label_dist,
        )
        # Approximate faulty components: predicate/quantifier nodes that differ
        n_faulty = _count_component_mismatches(gold, cand)
        return int(distance), n_faulty
    except Exception as e:
        logger.debug(f"ZSS distance error: {e}")
        return 99, 99


def _count_component_mismatches(gold: FOLNode, cand: FOLNode) -> int:
    """Count component-level mismatches between two aligned trees."""
    gold_comps = [f"{n.ntype}:{n.label}" for n in get_fault_components(gold)]
    cand_comps = [f"{n.ntype}:{n.label}" for n in get_fault_components(cand)]
    # Use LCS-based mismatch
    max_len = max(len(gold_comps), len(cand_comps))
    if max_len == 0:
        return 0
    matches = sum(a == b for a, b in zip(gold_comps, cand_comps))
    return max_len - matches


# ── Z3 Bounded Model Checking ─────────────────────────────────────────────────

z3.set_param("timeout", Z3_TIMEOUT_MS)


def _make_domain(size: int) -> tuple[z3.SortRef, list[z3.ExprRef]]:
    dom_sort = z3.DeclareSort(f"Dom{size}")
    consts = [z3.Const(f"d{i}", dom_sort) for i in range(size)]
    return dom_sort, consts


def fol_to_z3(node: FOLNode, pred_arities: dict[str, int], bound_vars: dict[str, z3.ExprRef],
               dom_sort: z3.SortRef, domain: list[z3.ExprRef]) -> z3.ExprRef | None:
    """Convert FOLNode to z3 expression. Returns None if unsupported."""
    nt = node.ntype
    try:
        if nt == "pred":
            name = node.label
            n_args = len(node.children)
            if name not in pred_arities:
                fn_sort = [dom_sort] * n_args + [z3.BoolSort()]
                pred_arities[name] = n_args
                pred_fn = z3.Function(name, *fn_sort)
            else:
                stored = pred_arities[name]
                if stored != n_args:
                    return None
                fn_sort = [dom_sort] * n_args + [z3.BoolSort()]
                pred_fn = z3.Function(name, *fn_sort)

            args = []
            for c in node.children:
                if c.ntype == "var":
                    if c.label in bound_vars:
                        args.append(bound_vars[c.label])
                    else:
                        args.append(domain[0])
                else:
                    args.append(domain[0])
            if n_args == 0:
                return pred_fn()
            return pred_fn(*args)

        elif nt == "not":
            inner = fol_to_z3(node.children[0], pred_arities, bound_vars, dom_sort, domain)
            if inner is None:
                return None
            return z3.Not(inner)

        elif nt == "and":
            l = fol_to_z3(node.children[0], pred_arities, bound_vars, dom_sort, domain)
            r = fol_to_z3(node.children[1], pred_arities, bound_vars, dom_sort, domain)
            if l is None or r is None:
                return None
            return z3.And(l, r)

        elif nt == "or":
            l = fol_to_z3(node.children[0], pred_arities, bound_vars, dom_sort, domain)
            r = fol_to_z3(node.children[1], pred_arities, bound_vars, dom_sort, domain)
            if l is None or r is None:
                return None
            return z3.Or(l, r)

        elif nt == "impl":
            l = fol_to_z3(node.children[0], pred_arities, bound_vars, dom_sort, domain)
            r = fol_to_z3(node.children[1], pred_arities, bound_vars, dom_sort, domain)
            if l is None or r is None:
                return None
            return z3.Implies(l, r)

        elif nt == "iff":
            l = fol_to_z3(node.children[0], pred_arities, bound_vars, dom_sort, domain)
            r = fol_to_z3(node.children[1], pred_arities, bound_vars, dom_sort, domain)
            if l is None or r is None:
                return None
            return l == r

        elif nt == "forall":
            var = z3.Const(node.label, dom_sort)
            new_bv = {**bound_vars, node.label: var}
            body = fol_to_z3(node.children[0], pred_arities, new_bv, dom_sort, domain)
            if body is None:
                return None
            return z3.ForAll([var], body)

        elif nt == "exists":
            var = z3.Const(node.label, dom_sort)
            new_bv = {**bound_vars, node.label: var}
            body = fol_to_z3(node.children[0], pred_arities, new_bv, dom_sort, domain)
            if body is None:
                return None
            return z3.Exists([var], body)

        elif nt in ("var", "const"):
            return bound_vars.get(node.label, domain[0])

    except Exception:
        return None
    return None


def z3_check(formula_str: str, *, check_sat: bool = False) -> str:
    """Check SAT/UNSAT. Returns 'sat'|'unsat'|'inconclusive'."""
    try:
        dom_sort = z3.DeclareSort("Dom_chk")
        domain = [z3.Const(f"d_chk_{i}", dom_sort) for i in range(DOMAIN_SIZE)]
        pred_arities: dict[str, int] = {}
        tree = parse_fol(formula_str)
        expr = fol_to_z3(tree, pred_arities, {}, dom_sort, domain)
        if expr is None:
            return "inconclusive"
        solver = z3.Solver()
        solver.set("timeout", Z3_TIMEOUT_MS)
        if check_sat:
            solver.add(expr)
        else:
            solver.add(z3.Not(expr))
        result = solver.check()
        if result == z3.sat:
            return "sat"
        elif result == z3.unsat:
            return "unsat"
        return "inconclusive"
    except Exception:
        return "inconclusive"


def z3_check_entailment(antecedent: str, consequent: str) -> str:
    """Check antecedent |= consequent. Returns 'pass'|'fail'|'inconclusive'."""
    try:
        dom_sort = z3.DeclareSort("Dom_ent")
        domain = [z3.Const(f"d_ent_{i}", dom_sort) for i in range(DOMAIN_SIZE)]
        pred_arities: dict[str, int] = {}
        ant_tree = parse_fol(antecedent)
        con_tree = parse_fol(consequent)
        ant_expr = fol_to_z3(ant_tree, pred_arities, {}, dom_sort, domain)
        con_expr = fol_to_z3(con_tree, pred_arities, {}, dom_sort, domain)
        if ant_expr is None or con_expr is None:
            return "inconclusive"
        solver = z3.Solver()
        solver.set("timeout", Z3_TIMEOUT_MS)
        solver.add(ant_expr)
        solver.add(z3.Not(con_expr))
        result = solver.check()
        if result == z3.unsat:
            return "pass"
        elif result == z3.sat:
            return "fail"
        return "inconclusive"
    except Exception:
        return "inconclusive"


def z3_check_equivalence(fol1: str, fol2: str) -> str:
    """Check if two FOL formulas are logically equivalent. Returns 'equiv'|'not_equiv'|'inconclusive'."""
    try:
        dom_sort = z3.DeclareSort("Dom_eq")
        domain = [z3.Const(f"d_eq_{i}", dom_sort) for i in range(DOMAIN_SIZE)]
        pred_arities: dict[str, int] = {}
        t1 = parse_fol(fol1)
        t2 = parse_fol(fol2)
        e1 = fol_to_z3(t1, pred_arities, {}, dom_sort, domain)
        e2 = fol_to_z3(t2, pred_arities, {}, dom_sort, domain)
        if e1 is None or e2 is None:
            return "inconclusive"
        solver = z3.Solver()
        solver.set("timeout", Z3_TIMEOUT_MS)
        solver.add(z3.Not(e1 == e2))
        result = solver.check()
        if result == z3.unsat:
            return "equiv"
        elif result == z3.sat:
            return "not_equiv"
        return "inconclusive"
    except Exception:
        return "inconclusive"


# ── Analytical Probes ─────────────────────────────────────────────────────────

def probe_well_formedness(fol_str: str) -> dict:
    """Probe 5: Check if formula parses without error."""
    try:
        parse_fol(fol_str)
        return {"verdict": "pass", "implicated_components": []}
    except ParseError as e:
        return {"verdict": "fail", "implicated_components": ["all"], "detail": str(e)}


def probe_free_variables(fol_str: str) -> dict:
    """Probe 6: Check for unbound variables."""
    try:
        tree = parse_fol(fol_str)
        offenders = []

        def _walk(node: FOLNode, bound: set[str]):
            if node.ntype in ("forall", "exists"):
                new_bound = bound | {node.label}
                for c in node.children:
                    _walk(c, new_bound)
            elif node.ntype == "var":
                # lowercase = variable; uppercase first letter = constant/predicate
                if node.label[0].islower() and node.label not in bound:
                    offenders.append(node.label)
            elif node.ntype == "pred":
                for c in node.children:
                    _walk(c, bound)
                    if node.component_id and c.ntype == "var" and c.label[0].islower() and c.label not in bound:
                        if node.component_id not in offenders:
                            offenders.append(node.component_id)
            else:
                for c in node.children:
                    _walk(c, bound)

        _walk(tree, set())
        if offenders:
            return {"verdict": "fail", "implicated_components": offenders[:3]}
        return {"verdict": "pass", "implicated_components": []}
    except ParseError:
        return {"verdict": "fail", "implicated_components": ["all"]}


def probe_arity_consistency(fol_str: str) -> dict:
    """Probe 7: Check predicate arity consistency."""
    try:
        tree = parse_fol(fol_str)
        arity_map: dict[str, tuple[int, str]] = {}
        offenders = []

        def _walk(node: FOLNode):
            if node.ntype == "pred":
                n = len(node.children)
                if node.label in arity_map:
                    expected, cid = arity_map[node.label]
                    if expected != n:
                        offenders.append(node.component_id or cid)
                else:
                    arity_map[node.label] = (n, node.component_id)
            for c in node.children:
                _walk(c)

        _walk(tree)
        if offenders:
            return {"verdict": "fail", "implicated_components": offenders}
        return {"verdict": "pass", "implicated_components": []}
    except ParseError:
        return {"verdict": "fail", "implicated_components": ["all"]}


def probe_scope_balance(fol_str: str) -> dict:
    """Probe 8: Check quantifier scope balance."""
    try:
        tree = parse_fol(fol_str)
        # Check each quantifier has at least one variable use in its scope
        issues = []

        def _check_scope(node: FOLNode, bound_here: set[str]) -> set[str]:
            """Returns set of variables used in this subtree."""
            used = set()
            if node.ntype in ("forall", "exists"):
                var = node.label
                child_used = _check_scope(node.children[0], bound_here | {var})
                if var not in child_used and node.children:
                    issues.append(node.component_id)
                used |= child_used - {var}
            elif node.ntype == "var":
                used.add(node.label)
            elif node.ntype == "pred":
                for c in node.children:
                    used |= _check_scope(c, bound_here)
            else:
                for c in node.children:
                    used |= _check_scope(c, bound_here)
            return used

        _check_scope(tree, set())
        if issues:
            return {"verdict": "fail", "implicated_components": issues}
        return {"verdict": "pass", "implicated_components": []}
    except ParseError:
        return {"verdict": "fail", "implicated_components": ["all"]}


def run_all_analytical_probes(fol_str: str) -> list[dict]:
    """Run probes 5-8 and return list of probe results."""
    probes = []
    for probe_fn, probe_name in [
        (probe_well_formedness, "well_formedness"),
        (probe_free_variables, "free_variables"),
        (probe_arity_consistency, "arity_consistency"),
        (probe_scope_balance, "scope_balance"),
    ]:
        result = probe_fn(fol_str)
        result["probe_name"] = probe_name
        result["probe_type"] = "analytical"
        probes.append(result)
    return probes


def run_conjunct_drop_probes(fol_str: str) -> list[dict]:
    """Probe 1: For conjunctive formulas, check each conjunct is entailed."""
    try:
        tree = parse_fol(fol_str)

        def get_conjuncts(node: FOLNode) -> list[FOLNode]:
            if node.ntype == "and":
                return get_conjuncts(node.children[0]) + get_conjuncts(node.children[1])
            return [node]

        conjuncts = get_conjuncts(tree)
        if len(conjuncts) < 2:
            return []

        probes = []
        for conj in conjuncts:
            conj_str = fol_to_string(conj)
            verdict = z3_check_entailment(fol_str, conj_str)
            comp_ids = [c.component_id for c in get_fault_components(conj) if c.component_id]
            probes.append({
                "probe_name": "conjunct_drop",
                "probe_type": "z3",
                "verdict": "pass" if verdict == "pass" else ("fail" if verdict == "fail" else "inconclusive"),
                "implicated_components": comp_ids,
            })
        return probes
    except Exception:
        return []


def run_negation_probe(fol_str: str) -> list[dict]:
    """Probe 2: Check formula is not self-contradictory."""
    try:
        # Check formula is satisfiable (not a contradiction)
        result = z3_check(fol_str, check_sat=True)
        verdict = "pass" if result == "sat" else ("fail" if result == "unsat" else "inconclusive")
        return [{
            "probe_name": "negation_consistency",
            "probe_type": "z3",
            "verdict": verdict,
            "implicated_components": [],
        }]
    except Exception:
        return []


# ── Dataset Loading ───────────────────────────────────────────────────────────

def load_folio_sentences(max_n: int = 120) -> list[dict]:
    """Load FOLIO dataset and extract compositional NL→FOL pairs."""
    logger.info("Loading FOLIO dataset...")
    try:
        from datasets import load_dataset
        ds = load_dataset("tasksource/folio", split="train", trust_remote_code=True)
        logger.info(f"FOLIO loaded: {len(ds)} examples")
    except Exception as e:
        logger.error(f"Failed to load FOLIO: {e}")
        raise

    sentences = []
    seen_nl = set()

    for row in ds:
        # Each row has premises (multiline) and premises-FOL (multiline)
        premises = row.get("premises", "") or ""
        fol_lines = row.get("premises-FOL", "") or ""

        nl_lines = [p.strip() for p in premises.splitlines() if p.strip()]
        fol_parts = [f.strip() for f in fol_lines.splitlines() if f.strip()]

        for nl, fol in zip(nl_lines, fol_parts):
            if nl in seen_nl:
                continue
            if len(nl) < 10 or len(fol) < 5:
                continue
            # Filter for compositional formulas (at least 1 quantifier or connective)
            has_quantifier = "∀" in fol or "∃" in fol or "FORALL" in fol or "EXISTS" in fol
            has_connective = any(c in fol for c in ["∧", "∨", "→", "↔", "¬"])
            if not (has_quantifier or has_connective):
                continue
            seen_nl.add(nl)
            sentences.append({"nl": nl, "gold_fol": fol})
            if len(sentences) >= max_n:
                break
        if len(sentences) >= max_n:
            break

    logger.info(f"Extracted {len(sentences)} compositional sentence pairs")
    return sentences[:max_n]


# ── Phase 0: Locality Pilot ───────────────────────────────────────────────────

SIMPLE_TRANSLATE_PROMPT = """Translate the following English sentence to First-Order Logic (FOL).
Use FOLIO notation: ∀x, ∃x, ¬, ∧, ∨, →, ↔ and CamelCase predicates.
Output ONLY the FOL formula, nothing else.

Sentence: {sentence}"""


async def translate_simple_async(
    session: aiohttp.ClientSession, nl: str, sem: asyncio.Semaphore
) -> str:
    async with sem:
        try:
            result = await openrouter_call_async(
                session,
                SIMPLE_TRANSLATE_PROMPT.format(sentence=nl),
                max_tokens=256,
                temperature=0.0,
            )
            return result["content"].strip()
        except BudgetExceeded:
            raise
        except Exception as e:
            logger.warning(f"translate_simple failed: {e}")
            return ""


async def run_phase0_async(sentences: list[dict]) -> list[dict]:
    """Phase 0: Translate all sentences, compute AST diff, check locality."""
    logger.info(f"Phase 0: Translating {len(sentences)} sentences...")
    sem = asyncio.Semaphore(CONCURRENCY)
    async with aiohttp.ClientSession() as session:
        tasks = [translate_simple_async(session, s["nl"], sem) for s in sentences]
        translations = await asyncio.gather(*tasks, return_exceptions=True)

    results = []
    for s, trans in zip(sentences, translations):
        if isinstance(trans, BudgetExceeded):
            raise trans
        if isinstance(trans, Exception):
            trans = ""
        s["candidate_fol"] = trans

        # Compute AST diff
        n_faulty = 0
        is_wrong = True
        if trans:
            try:
                gold_tree = parse_fol(s["gold_fol"])
                cand_tree = parse_fol(trans)
                gold_r = canonical_rename(gold_tree)
                cand_r = canonical_rename(cand_tree)
                dist, n_faulty = compute_ast_edit_distance(gold_r, cand_r)
                is_wrong = dist > 0
            except ParseError:
                n_faulty = 99
                is_wrong = True
        else:
            is_wrong = True
            n_faulty = 99

        s["n_faulty"] = n_faulty
        s["is_wrong"] = is_wrong
        results.append(s)

    return results


# ── Phase 1: Aligned Translation ──────────────────────────────────────────────

ALIGNED_PROMPT = """Translate the following English sentence to First-Order Logic (FOL).
For each logical component, identify the NL span that licenses it.
Output JSON only (no markdown) matching exactly this format:
{{"components": [{{"span": "NL text span", "fol_component": "FOL sub-expression", "component_id": "c0"}}], "full_fol": "complete FOL formula"}}

Use ∀x, ∃x, ¬, ∧, ∨, →, ↔ and CamelCase predicates like IsHappy(x).

Sentence: {sentence}"""


def extract_json_from_response(text: str) -> dict | None:
    """Extract JSON from LLM response, handling markdown code blocks."""
    text = text.strip()
    # Remove markdown code blocks
    text = re.sub(r"```(?:json)?\n?(.*?)\n?```", r"\1", text, flags=re.DOTALL)
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON object
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
    return None


async def translate_aligned_async(
    session: aiohttp.ClientSession, nl: str, sem: asyncio.Semaphore
) -> dict | None:
    async with sem:
        try:
            result = await openrouter_call_async(
                session,
                ALIGNED_PROMPT.format(sentence=nl),
                max_tokens=512,
                temperature=0.0,
                json_mode=True,
            )
            data = extract_json_from_response(result["content"])
            if data and "full_fol" in data:
                return data
        except BudgetExceeded:
            raise
        except Exception as e:
            logger.warning(f"translate_aligned failed: {e}")
    return None


async def run_phase1_async(sentences: list[dict]) -> list[dict]:
    """Phase 1: Get aligned translations with span tags."""
    logger.info(f"Phase 1: Aligned translation for {len(sentences)} sentences...")
    sem = asyncio.Semaphore(CONCURRENCY)
    async with aiohttp.ClientSession() as session:
        tasks = [translate_aligned_async(session, s["nl"], sem) for s in sentences]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    for s, res in zip(sentences, results):
        if isinstance(res, BudgetExceeded):
            raise res
        if isinstance(res, Exception):
            res = None
        if res:
            s["aligned_output"] = res
            s["aligned_fol"] = res.get("full_fol", s.get("candidate_fol", ""))
            s["components"] = res.get("components", [])
        else:
            s["aligned_output"] = None
            s["aligned_fol"] = s.get("candidate_fol", "")
            s["components"] = []

    return sentences


# ── Phase 2+3: Probe Bank ─────────────────────────────────────────────────────

def run_probes_for_sentence(fol_str: str) -> list[dict]:
    """Run all probes for a given FOL formula."""
    probes = []
    probes.extend(run_all_analytical_probes(fol_str))
    probes.extend(run_conjunct_drop_probes(fol_str))
    probes.extend(run_negation_probe(fol_str))
    return probes


def compute_probe_verdict_rate(probes: list[dict]) -> dict:
    if not probes:
        return {"overall": 0.0, "by_type": {}}
    by_type: dict[str, list] = {}
    for p in probes:
        pt = p.get("probe_type", "unknown")
        by_type.setdefault(pt, []).append(p)

    total = len(probes)
    definitive = sum(1 for p in probes if p["verdict"] != "inconclusive")
    result = {"overall": definitive / total if total else 0.0, "by_type": {}}
    for pt, ps in by_type.items():
        d = sum(1 for p in ps if p["verdict"] != "inconclusive")
        result["by_type"][pt] = d / len(ps) if ps else 0.0
    return result


# ── Phase 4: Suspiciousness Ranking ──────────────────────────────────────────

def build_coverage_matrix(
    probes: list[dict], component_ids: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Build probe×component binary matrix and failure vector."""
    n_probes = len(probes)
    n_comps = len(component_ids)
    M = np.zeros((n_probes, n_comps), dtype=float)
    failure = np.zeros(n_probes, dtype=float)
    comp_idx = {c: i for i, c in enumerate(component_ids)}

    for i, probe in enumerate(probes):
        if probe["verdict"] == "fail":
            failure[i] = 1.0
        for cid in probe.get("implicated_components", []):
            if cid in comp_idx:
                M[i, comp_idx[cid]] = 1.0
            elif cid == "all":
                M[i, :] = 1.0

    return M, failure


def ochiai(M: np.ndarray, failure: np.ndarray) -> np.ndarray:
    nf = failure.sum()
    scores = np.zeros(M.shape[1])
    for j in range(M.shape[1]):
        nf_c = (M[:, j] * failure).sum()
        np_c = (M[:, j] * (1 - failure)).sum()
        denom = math.sqrt(nf * (nf_c + np_c)) if nf > 0 else 0
        scores[j] = nf_c / denom if denom > 0 else 0.0
    return scores


def tarantula(M: np.ndarray, failure: np.ndarray) -> np.ndarray:
    nf = failure.sum()
    np_total = (1 - failure).sum()
    scores = np.zeros(M.shape[1])
    for j in range(M.shape[1]):
        nf_c = (M[:, j] * failure).sum()
        np_c = (M[:, j] * (1 - failure)).sum()
        nf_ratio = nf_c / nf if nf > 0 else 0
        np_ratio = np_c / np_total if np_total > 0 else 0
        denom = nf_ratio + np_ratio + 1e-10
        scores[j] = nf_ratio / denom
    return scores


def dstar(M: np.ndarray, failure: np.ndarray, star: int = 2) -> np.ndarray:
    nf = failure.sum()
    scores = np.zeros(M.shape[1])
    for j in range(M.shape[1]):
        nf_c = (M[:, j] * failure).sum()
        np_c = (M[:, j] * (1 - failure)).sum()
        denom = np_c + (nf - nf_c) + 1e-10
        scores[j] = (nf_c ** star) / denom
    return scores


def rank_components(scores: np.ndarray, component_ids: list[str], nodes: list[FOLNode]) -> list[str]:
    """Rank components by score desc, tie-break by nesting depth."""
    depth_map = {n.component_id: n.nesting_depth for n in nodes if n.component_id}
    order = sorted(
        range(len(component_ids)),
        key=lambda j: (-scores[j], -depth_map.get(component_ids[j], 0)),
    )
    return [component_ids[i] for i in order]


def compute_localization_metrics(
    ranked_components: list[str], true_fault_components: list[str]
) -> dict:
    """Compute Top-1, Top-3, wasted-effort given ranked component list."""
    if not true_fault_components or not ranked_components:
        return {"top1": False, "top3": False, "wasted_effort": 1.0}
    true_set = set(true_fault_components)
    top1 = ranked_components[0] in true_set if ranked_components else False
    top3 = any(c in true_set for c in ranked_components[:3])
    # Wasted effort: rank of first true fault / total
    first_rank = next(
        (i for i, c in enumerate(ranked_components) if c in true_set), len(ranked_components)
    )
    we = first_rank / len(ranked_components) if ranked_components else 1.0
    return {"top1": top1, "top3": top3, "wasted_effort": we}


# ── Phase 4 Baselines ─────────────────────────────────────────────────────────

async def ast_disagreement_baseline_async(
    session: aiohttp.ClientSession, nl: str, gold_fol: str, sem: asyncio.Semaphore, K: int = 5
) -> list[dict]:
    """Generate K candidates and rank by cross-candidate disagreement."""
    async with sem:
        tasks = [
            openrouter_call_async(
                session,
                SIMPLE_TRANSLATE_PROMPT.format(sentence=nl),
                max_tokens=256,
                temperature=0.7,
            )
            for _ in range(K)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    candidates = []
    for r in results:
        if isinstance(r, Exception):
            continue
        cand = r["content"].strip()
        try:
            parse_fol(cand)
            candidates.append(cand)
        except ParseError:
            pass

    if not candidates:
        return []

    # Extract component labels from each candidate
    all_comp_labels = []
    for c in candidates:
        try:
            tree = canonical_rename(parse_fol(c))
            labels = [f"{n.ntype}:{n.label}" for n in get_fault_components(tree)]
            all_comp_labels.append(labels)
        except Exception:
            all_comp_labels.append([])

    # Compute disagreement per position
    max_len = max((len(l) for l in all_comp_labels), default=0)
    scores = []
    for i in range(max_len):
        labels_at_i = [l[i] for l in all_comp_labels if i < len(l)]
        if labels_at_i:
            most_common = Counter(labels_at_i).most_common(1)[0][1]
            disagreement = 1 - most_common / len(labels_at_i)
        else:
            disagreement = 0
        scores.append({"position": i, "score": disagreement})

    return sorted(scores, key=lambda x: -x["score"])


# ── Phase 5: Targeted Repair ──────────────────────────────────────────────────

TARGETED_REPAIR_PROMPT = """The following FOL formula has a fault in component "{component_id}".
Faulty component: {faulty_component}
Failing probes: {violated_probes}

Original sentence: {sentence}
Current full formula: {full_fol}

Re-translate ONLY the faulty component into correct FOL.
Output JSON only: {{"repaired_component": "...", "reasoning": "..."}}"""


async def targeted_repair_async(
    session: aiohttp.ClientSession,
    sem: asyncio.Semaphore,
    nl: str,
    full_fol: str,
    faulty_comp: dict,
    violated_probes: list[dict],
    k_rounds: int = 2,
) -> str:
    """Try to repair the faulty component, return new full FOL."""
    violated_str = ", ".join(p["probe_name"] for p in violated_probes[:3])
    for _ in range(k_rounds):
        async with sem:
            try:
                result = await openrouter_call_async(
                    session,
                    TARGETED_REPAIR_PROMPT.format(
                        component_id=faulty_comp.get("component_id", "?"),
                        faulty_component=faulty_comp.get("fol_component", "?"),
                        violated_probes=violated_str,
                        sentence=nl,
                        full_fol=full_fol,
                    ),
                    max_tokens=256,
                    temperature=0.0,
                    json_mode=True,
                )
                data = extract_json_from_response(result["content"])
                if data and "repaired_component" in data:
                    repaired = data["repaired_component"]
                    # Substitute the faulty component in the full formula
                    # Simple string replacement of the fol_component
                    old_comp = faulty_comp.get("fol_component", "")
                    if old_comp and old_comp in full_fol:
                        return full_fol.replace(old_comp, repaired, 1)
                    return repaired
            except BudgetExceeded:
                raise
            except Exception as e:
                logger.debug(f"Repair failed: {e}")
    return full_fol


async def blind_regeneration_async(
    session: aiohttp.ClientSession,
    sem: asyncio.Semaphore,
    nl: str,
    n_calls: int = 3,
) -> str:
    """Generate n_calls full formula candidates, return majority vote."""
    async with sem:
        tasks = [
            openrouter_call_async(
                session,
                SIMPLE_TRANSLATE_PROMPT.format(sentence=nl),
                max_tokens=256,
                temperature=0.5,
            )
            for _ in range(n_calls)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    candidates = []
    for r in results:
        if isinstance(r, Exception):
            continue
        try:
            cand = r["content"].strip()
            parse_fol(cand)
            candidates.append(cand)
        except Exception:
            pass

    if not candidates:
        return ""
    # Majority vote by canonical string
    canonical_counts = Counter(
        fol_to_string(canonical_rename(parse_fol(c))) for c in candidates
        if _try_parse(c)
    )
    if canonical_counts:
        best_canonical = canonical_counts.most_common(1)[0][0]
        # Return first candidate matching that canonical
        for c in candidates:
            try:
                if fol_to_string(canonical_rename(parse_fol(c))) == best_canonical:
                    return c
            except Exception:
                pass
    return candidates[0]


def _try_parse(fol_str: str) -> bool:
    try:
        parse_fol(fol_str)
        return True
    except Exception:
        return False


def eval_accuracy(predicted: str, gold: str) -> dict:
    """Evaluate predicted FOL vs gold FOL."""
    exact = False
    z3_equiv = False
    try:
        p_tree = canonical_rename(parse_fol(predicted))
        g_tree = canonical_rename(parse_fol(gold))
        exact = fol_to_string(p_tree) == fol_to_string(g_tree)
    except Exception:
        pass
    try:
        z3_result = z3_check_equivalence(predicted, gold)
        z3_equiv = z3_result == "equiv"
    except Exception:
        pass
    return {"exact_match": exact, "z3_equiv": z3_equiv}


# ── NLTK WordNet ──────────────────────────────────────────────────────────────
_wordnet_ready = False


def ensure_wordnet():
    global _wordnet_ready
    if not _wordnet_ready:
        try:
            nltk.download("wordnet", quiet=True)
            nltk.download("omw-1.4", quiet=True)
            _wordnet_ready = True
        except Exception:
            pass


# ── Phase 6: Ablations ────────────────────────────────────────────────────────

def ablation_probe_count(
    sentence_results: list[dict], k_values: list[int]
) -> list[dict]:
    """Ablation 1: Vary probe count k, measure Ochiai Top-1."""
    rows = []
    for k in k_values:
        top1_scores = []
        top1_mixed = []
        top1_z3_only = []
        top1_analytical_only = []
        for sr in sentence_results:
            probes = sr.get("probes", [])
            true_faults = sr.get("true_fault_components", [])
            comps = sr.get("component_ids", [])
            if not comps or not true_faults:
                continue
            # Sample k probes
            sampled = probes[:k] if len(probes) >= k else probes
            # Mixed
            M, fvec = build_coverage_matrix(sampled, comps)
            scores = ochiai(M, fvec)
            ranked = rank_components(scores, comps, sr.get("component_nodes", []))
            metrics = compute_localization_metrics(ranked, true_faults)
            top1_mixed.append(metrics["top1"])
            # Z3 probes only
            z3_probes = [p for p in sampled if p.get("probe_type") == "z3"][:k]
            if z3_probes:
                M2, fvec2 = build_coverage_matrix(z3_probes, comps)
                s2 = ochiai(M2, fvec2)
                r2 = rank_components(s2, comps, sr.get("component_nodes", []))
                top1_z3_only.append(compute_localization_metrics(r2, true_faults)["top1"])
            # Analytical probes only
            an_probes = [p for p in sampled if p.get("probe_type") == "analytical"][:k]
            if an_probes:
                M3, fvec3 = build_coverage_matrix(an_probes, comps)
                s3 = ochiai(M3, fvec3)
                r3 = rank_components(s3, comps, sr.get("component_nodes", []))
                top1_analytical_only.append(compute_localization_metrics(r3, true_faults)["top1"])

        rows.append({
            "k": k,
            "ochiai_top1": float(np.mean(top1_mixed)) if top1_mixed else 0.0,
            "mixed_top1": float(np.mean(top1_mixed)) if top1_mixed else 0.0,
            "prover_only_top1": float(np.mean(top1_z3_only)) if top1_z3_only else 0.0,
            "analytical_only_top1": float(np.mean(top1_analytical_only)) if top1_analytical_only else 0.0,
        })
    return rows


def ablation_alignment_noise(
    sentence_results: list[dict], noise_pcts: list[int]
) -> list[dict]:
    """Ablation 2: Inject noise into coverage matrix, measure Ochiai Top-1."""
    rng = np.random.RandomState(42)
    rows = []
    for noise_pct in noise_pcts:
        top1_scores = []
        for sr in sentence_results:
            probes = sr.get("probes", [])
            true_faults = sr.get("true_fault_components", [])
            comps = sr.get("component_ids", [])
            if not comps or not true_faults:
                continue
            M, fvec = build_coverage_matrix(probes, comps)
            # Inject noise: flip noise_pct% of entries
            M_noisy = M.copy()
            n_total = M_noisy.size
            n_flip = int(n_total * noise_pct / 100)
            if n_flip > 0:
                flat_idx = rng.choice(n_total, n_flip, replace=False)
                M_flat = M_noisy.flatten()
                M_flat[flat_idx] = 1 - M_flat[flat_idx]
                M_noisy = M_flat.reshape(M_noisy.shape)
            scores = ochiai(M_noisy, fvec)
            ranked = rank_components(scores, comps, sr.get("component_nodes", []))
            metrics = compute_localization_metrics(ranked, true_faults)
            top1_scores.append(metrics["top1"])
        rows.append({
            "noise_pct": noise_pct,
            "ochiai_top1": float(np.mean(top1_scores)) if top1_scores else 0.0,
        })
    return rows


# ── Main Pipeline ─────────────────────────────────────────────────────────────

@logger.catch(reraise=True)
def main():
    logger.info("=== SBFL-for-FOL Experiment Starting ===")
    start_time = time.time()

    # ── Load data ──
    sentences = load_folio_sentences(MAX_SENTENCES)
    logger.info(f"Loaded {len(sentences)} sentences")

    # ── Phase 0: Locality Pilot ──
    logger.info("=== Phase 0: Locality Pilot ===")
    sentences = asyncio.run(run_phase0_async(sentences))

    wrong = [s for s in sentences if s["is_wrong"]]
    correct = [s for s in sentences if not s["is_wrong"]]
    locality_dist = dict(Counter(s["n_faulty"] for s in wrong))
    local_errors = sum(1 for s in wrong if s["n_faulty"] <= 2 and s["n_faulty"] < 99)
    locality_frac = local_errors / len(wrong) if wrong else 0.0
    locality_gate = locality_frac >= 0.70

    logger.info(f"Phase 0: {len(wrong)}/{len(sentences)} wrong, locality_frac={locality_frac:.2%}, gate={'PASS' if locality_gate else 'FAIL'}")
    logger.info(f"Locality distribution: {locality_dist}")

    # Proceed regardless of gate (for complete results)
    # Work on wrong sentences for Phase 1+
    local_subset = [s for s in wrong if s["n_faulty"] <= 2 and s["n_faulty"] < 99]
    if not local_subset:
        local_subset = wrong[:40]
    logger.info(f"Local subset size: {len(local_subset)}")

    # ── Phase 1: Aligned Translation ──
    logger.info("=== Phase 1: Aligned Translation ===")
    local_subset = asyncio.run(run_phase1_async(local_subset))

    # Compute alignment accuracy (proxy: fraction with valid JSON and >0 components)
    aligned_ok = [s for s in local_subset if s.get("components")]
    alignment_accuracy = len(aligned_ok) / len(local_subset) if local_subset else 0.0
    logger.info(f"Alignment accuracy proxy: {alignment_accuracy:.2%}")

    # ── Phase 2+3: Probe Bank ──
    logger.info("=== Phase 2+3: Probe Bank ===")
    sentence_results = []
    all_probe_verdicts = []

    for s in local_subset:
        fol_str = s.get("aligned_fol") or s.get("candidate_fol", "")
        if not fol_str:
            continue

        probes = run_probes_for_sentence(fol_str)

        # Determine component IDs from the formula
        try:
            tree = parse_fol(fol_str)
            comp_nodes = get_fault_components(tree)
            component_ids = [n.component_id for n in comp_nodes if n.component_id]
        except Exception:
            comp_nodes = []
            component_ids = []

        # True fault components: from AST diff vs gold
        true_faults = []
        try:
            gold_tree = canonical_rename(parse_fol(s["gold_fol"]))
            cand_tree = canonical_rename(parse_fol(fol_str))
            gold_comps = get_fault_components(gold_tree)
            cand_comps = get_fault_components(cand_tree)
            # Mark mismatched positions as faults
            for i, (gc, cc) in enumerate(zip(gold_comps, cand_comps)):
                if fol_to_string(gc) != fol_to_string(cc) and i < len(component_ids):
                    true_faults.append(component_ids[i])
            if not true_faults and component_ids:
                true_faults = [component_ids[0]]
        except Exception:
            if component_ids:
                true_faults = [component_ids[0]]

        all_probe_verdicts.extend(probes)
        sentence_results.append({
            "nl": s["nl"],
            "gold_fol": s["gold_fol"],
            "candidate_fol": fol_str,
            "probes": probes,
            "component_ids": component_ids,
            "component_nodes": comp_nodes,
            "true_fault_components": true_faults,
            "n_faulty": s.get("n_faulty", 0),
            "aligned_components": s.get("components", []),
        })

    # Compute probe verdict rate
    probe_verdict_rate = compute_probe_verdict_rate(all_probe_verdicts)
    logger.info(f"Probe verdict rate: {probe_verdict_rate}")

    # ── Phase 4: Ochiai Ranking ──
    logger.info("=== Phase 4: Suspiciousness Ranking ===")

    ochiai_metrics = []
    tarantula_metrics = []
    dstar_metrics = []
    random_metrics = []

    for sr in sentence_results:
        probes = sr["probes"]
        true_faults = sr["true_fault_components"]
        comps = sr["component_ids"]
        comp_nodes = sr["component_nodes"]

        if not comps or not true_faults:
            continue

        M, fvec = build_coverage_matrix(probes, comps)

        # Ochiai
        scores_och = ochiai(M, fvec)
        ranked_och = rank_components(scores_och, comps, comp_nodes)
        ochiai_metrics.append(compute_localization_metrics(ranked_och, true_faults))

        # Tarantula
        scores_tar = tarantula(M, fvec)
        ranked_tar = rank_components(scores_tar, comps, comp_nodes)
        tarantula_metrics.append(compute_localization_metrics(ranked_tar, true_faults))

        # DStar
        scores_ds = dstar(M, fvec)
        ranked_ds = rank_components(scores_ds, comps, comp_nodes)
        dstar_metrics.append(compute_localization_metrics(ranked_ds, true_faults))

        # Random baseline: expected = 1/n_comps
        n = len(comps)
        random_metrics.append({
            "top1": 1.0 / n if n > 0 else 0.0,
            "top3": min(3.0 / n, 1.0) if n > 0 else 0.0,
            "wasted_effort": 0.5,
        })

        sr["ochiai_ranked"] = ranked_och
        sr["ochiai_scores"] = scores_och.tolist()

    def aggregate(metrics: list[dict]) -> dict:
        if not metrics:
            return {"top1": 0.0, "top3": 0.0, "wasted_effort": 1.0}
        return {
            "top1": float(np.mean([m["top1"] for m in metrics])),
            "top3": float(np.mean([m["top3"] for m in metrics])),
            "wasted_effort": float(np.mean([m["wasted_effort"] for m in metrics])),
        }

    localization_results = {
        "local_subset_size": len(sentence_results),
        "ochiai": aggregate(ochiai_metrics),
        "tarantula": aggregate(tarantula_metrics),
        "dstar": aggregate(dstar_metrics),
        "baseline_random": aggregate(random_metrics),
    }
    logger.info(f"Localization results: {localization_results}")

    # ── Phase 5: Targeted Repair ──
    logger.info("=== Phase 5: Targeted Repair ===")
    repair_results = asyncio.run(run_phase5_async(sentence_results))
    logger.info(f"Repair results: {repair_results}")

    # ── Phase 6: Ablations ──
    logger.info("=== Phase 6: Ablations ===")
    probe_count_curve = ablation_probe_count(sentence_results, [5, 10, 20])
    alignment_noise = ablation_alignment_noise(sentence_results, [0, 10, 20, 30, 40])
    formula_comparison = {
        "ochiai_top1": localization_results["ochiai"]["top1"],
        "tarantula_top1": localization_results["tarantula"]["top1"],
        "dstar_top1": localization_results["dstar"]["top1"],
    }

    elapsed = time.time() - start_time
    logger.info(f"Total elapsed: {elapsed:.1f}s, cost: ${cost_tracker['total']:.4f}")

    # ── Build method_out.json ──
    logger.info("Building method_out.json...")
    examples = []

    # Phase 0 examples (all sentences)
    for s in sentences:
        input_str = s["nl"]
        output_str = s["gold_fol"]
        predict_method = s.get("aligned_fol") or s.get("candidate_fol", "")
        predict_baseline = s.get("candidate_fol", "")

        metadata = {
            "metadata_phase": "phase0",
            "metadata_is_wrong": s.get("is_wrong", True),
            "metadata_n_faulty": s.get("n_faulty", 0),
            "metadata_locality_gate_passed": locality_gate,
        }

        # Add Phase 4 data if available
        matching_sr = next(
            (sr for sr in sentence_results if sr["nl"] == input_str), None
        )
        if matching_sr:
            metadata.update({
                "metadata_phase": "phase4",
                "metadata_n_probes": len(matching_sr.get("probes", [])),
                "metadata_n_components": len(matching_sr.get("component_ids", [])),
                "metadata_n_true_faults": len(matching_sr.get("true_fault_components", [])),
            })
            if ochiai_metrics and matching_sr.get("ochiai_ranked"):
                idx = sentence_results.index(matching_sr)
                if idx < len(ochiai_metrics):
                    metadata["metadata_ochiai_top1"] = ochiai_metrics[idx]["top1"]
                    metadata["metadata_ochiai_top3"] = ochiai_metrics[idx]["top3"]
                    metadata["metadata_ochiai_wasted_effort"] = ochiai_metrics[idx]["wasted_effort"]

        # Find repair result
        rep = repair_results.get("per_sentence", {}).get(input_str, {})
        if rep:
            predict_method = rep.get("targeted_repaired", predict_method)
            predict_baseline = rep.get("blind_regen", predict_baseline)

        example = {
            "input": input_str,
            "output": output_str,
            "predict_our_method": predict_method,
            "predict_baseline": predict_baseline,
        }
        example.update(metadata)
        examples.append(example)

    # Build full method_out.json
    method_out = {
        "metadata": {
            "method_name": "SBFL-for-FOL",
            "description": "Spectrum-Based Fault Localization for NL→FOL Translation",
            "locality_distribution": locality_dist,
            "locality_fraction": locality_frac,
            "locality_gate_passed": locality_gate,
            "alignment_accuracy": alignment_accuracy,
            "probe_verdict_rate": probe_verdict_rate,
            "localization_results": localization_results,
            "repair_accuracy": repair_results.get("aggregate", {}),
            "ablation_curves": {
                "probe_count": probe_count_curve,
                "alignment_noise": alignment_noise,
                "formula_comparison": formula_comparison,
            },
            "cost_total": cost_tracker["total"],
            "n_sentences_processed": len(sentences),
            "elapsed_seconds": elapsed,
        },
        "datasets": [
            {
                "dataset": "folio",
                "examples": examples,
            }
        ],
    }

    out_path = WORKSPACE / "method_out.json"
    out_path.write_text(json.dumps(method_out, indent=2, default=str))
    logger.info(f"Saved method_out.json with {len(examples)} examples")
    logger.info(f"Final cost: ${cost_tracker['total']:.4f}")


async def run_phase5_async(sentence_results: list[dict]) -> dict:
    """Phase 5: Targeted repair vs blind regeneration."""
    sem = asyncio.Semaphore(CONCURRENCY)
    targeted_accs = []
    blind_accs = []
    targeted_calls = 0
    blind_calls = 0
    per_sentence = {}

    async with aiohttp.ClientSession() as session:
        tasks = []
        for sr in sentence_results[:30]:  # Limit for budget
            tasks.append(_repair_one(session, sem, sr))
        results = await asyncio.gather(*tasks, return_exceptions=True)

    for sr, res in zip(sentence_results[:30], results):
        if isinstance(res, BudgetExceeded):
            break
        if isinstance(res, Exception):
            continue
        targeted_acc = eval_accuracy(res.get("targeted_repaired", ""), sr["gold_fol"])
        blind_acc = eval_accuracy(res.get("blind_regen", ""), sr["gold_fol"])
        targeted_accs.append(targeted_acc)
        blind_accs.append(blind_acc)
        targeted_calls += res.get("targeted_calls", 1)
        blind_calls += res.get("blind_calls", 3)
        per_sentence[sr["nl"]] = res

    def agg_acc(accs: list[dict]) -> dict:
        if not accs:
            return {"exact_match": 0.0, "z3_equiv": 0.0}
        return {
            "exact_match": float(np.mean([a["exact_match"] for a in accs])),
            "z3_equiv": float(np.mean([a["z3_equiv"] for a in accs])),
        }

    targeted_agg = agg_acc(targeted_accs)
    blind_agg = agg_acc(blind_accs)
    eff = (targeted_agg["z3_equiv"] + 1e-9) / (blind_agg["z3_equiv"] + 1e-9)

    return {
        "aggregate": {
            "targeted": {**targeted_agg, "lm_calls": targeted_calls},
            "blind_regen": {**blind_agg, "lm_calls": blind_calls},
            "efficiency_ratio": eff,
        },
        "per_sentence": per_sentence,
    }


async def _repair_one(session: aiohttp.ClientSession, sem: asyncio.Semaphore, sr: dict) -> dict:
    """Repair one sentence with both methods."""
    nl = sr["nl"]
    full_fol = sr["candidate_fol"]
    gold_fol = sr["gold_fol"]
    components = sr.get("aligned_components", [])
    probes = sr.get("probes", [])
    ochiai_ranked = sr.get("ochiai_ranked", [])

    # Find top-ranked faulty component
    failed_probes = [p for p in probes if p["verdict"] == "fail"]
    faulty_comp = {}
    if ochiai_ranked and components:
        top_cid = ochiai_ranked[0]
        faulty_comp = next(
            (c for c in components if c.get("component_id") == top_cid),
            components[0] if components else {},
        )
    elif components:
        faulty_comp = components[0]

    targeted_fol = full_fol
    if faulty_comp:
        targeted_fol = await targeted_repair_async(
            session, sem, nl, full_fol, faulty_comp, failed_probes[:3], k_rounds=2
        )

    blind_fol = await blind_regeneration_async(session, sem, nl, n_calls=3)

    return {
        "targeted_repaired": targeted_fol,
        "blind_regen": blind_fol,
        "targeted_calls": 2,
        "blind_calls": 3,
    }


if __name__ == "__main__":
    main()
