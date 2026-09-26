"""Answer extraction and correctness checking, used for training rewards and for scoring.
Run directly for its self-tests."""
from __future__ import annotations

import ast
import math
import re
import sys

SCRIPT_VERSION = "csm_verify/2.0"

# tolerance per dataset: ChartQA's published protocol is 5% relative
TOLERANCE = {"chartqa": 0.05, "geometry3k": 1e-3,
             # same numeric regime as geometry3k: angles, lengths, short decimals
             "mathvista": 1e-3, "mathverse": 1e-3, "default": 1e-6}


# --------------------------------------------------------------- extraction
_BOXED = re.compile(r"\\boxed\s*\{")
_ANSWER_TAG = re.compile(
    r"(?:final\s+answer|answer)\s*(?:is)?\s*[:=]\s*(.+?)(?:\n|$)", re.IGNORECASE)
_ANSWER_XML = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)


def _match_brace(s: str, open_idx: int) -> str | None:
    """Return the contents of the {...} whose '{' is at open_idx, or None."""
    depth = 0
    for i in range(open_idx, len(s)):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                return s[open_idx + 1:i]
    return None


def extract_answer(completion: str) -> str:
    """Pull the model's final answer out of a completion.

    Priority: \\boxed{...} > <answer>...</answer> > 'Answer: ...' > last
    non-empty line. The LAST occurrence wins, since models restate and refine.
    Returns "" when nothing looks like an answer, which scores as incorrect --
    the conservative direction.
    """
    if not completion:
        return ""
    text = completion.strip()

    # \boxed{...}: take the last one
    # Take the last NON-EMPTY box. A trailing empty \boxed{}, which models emit
    # when they start a box and stop, used to win last-occurrence and erase a
    # real answer given earlier.
    last, last_nonempty = None, None
    for m in _BOXED.finditer(text):
        inner = _match_brace(text, m.end() - 1)
        if inner is not None:
            last = inner
            if inner.strip():
                last_nonempty = inner
    if last_nonempty is not None:
        return last_nonempty.strip()
    if last is not None:
        return last.strip()

    m = list(_ANSWER_XML.finditer(text))
    if m:
        return m[-1].group(1).strip()

    m = list(_ANSWER_TAG.finditer(text))
    if m:
        return m[-1].group(1).strip()

    for line in reversed(text.splitlines()):
        if line.strip():
            return line.strip()
    return ""


# -------------------------------------------------------------- normalizing
# Order matters. Argument-consuming rules must run before the bare-macro rules,
# and the escaped dollar before the bare dollar, or a lone backslash survives and
# poisons the numeric parse.
_LATEX_STRIP = [
    (re.compile(r"\\left|\\right"), ""),
    # Degree marks. The unicode sign and the WORDS "degrees/deg" were already
    # handled, but the LaTeX spellings were not, so 62^\circ against gold 62
    # scored WRONG. Measured on 57,600 stored rollouts this flipped 237 verdicts
    # and hit the untrained base arm ~20x harder than the trained seeds, because
    # the verifier IS the training reward and so taught the seeds to drop it.
    (re.compile(r"\^\s*\{?\s*\\circ\s*\}?|\\circ\b|\\degree\b"), ""),
    # Prime marks (feet, arc-minutes) are decoration in the same way.
    (re.compile(r"\^\s*\{?\s*\\prime\s*\}?|\\prime\b"), ""),
    # Argument-taking layout macros: drop the macro AND its argument.
    (re.compile(r"\\(?:hspace|phantom|vspace)\s*\{[^}]*\}"), ""),
    # Presentation-only wrappers: keep the argument, drop the wrapper.
    (re.compile(r"\\(?:text|mathrm|mathbf|textbf|mathit|textit|bm|boldsymbol"
                r"|mbox|operatorname|overline|widehat|overrightarrow)\s*\{([^}]*)\}"), r"\1"),
    (re.compile(r"\\overset\s*\{[^}]*\}\s*\{([^}]*)\}"), r"\1"),
    # Named spacing macros, then the punctuation-spacing forms.
    (re.compile(r"\\(?:quad|qquad|,|;|!|:)\b"), " "),
    (re.compile(r"\\[,;!:\s]"), " "),
    # Geometry labels and approximation relations are noise around the value.
    (re.compile(r"\\angle\b|\\measuredangle\b"), " "),
    (re.compile(r"\\(?:approx|simeq|cong|sim)\b"), " "),
    (re.compile(r"\\displaystyle"), ""),
    (re.compile(r"\\\$"), ""),          # escaped dollar, BEFORE the bare one
    (re.compile(r"\$"), ""),
    (re.compile(r"\\%"), "%"),
]

# A unit must swallow its OWN exponent. Stripping "cm" out of "37 cm^2" used to
# leave "37 ^2", which to_number then evaluated as 37 squared = 1369.
_UNITS = re.compile(
    r"\b(cm|mm|km|m|in|ft|inches|feet|units?|degrees?|deg|sq|square|"
    r"dollars?|percent|people|years?)\b\.?"
    r"(\s*\^\s*\{?\s*\d+\s*\}?)?", re.IGNORECASE)


def normalize_text(s: str) -> str:
    """Lowercase, de-LaTeX, drop separators/units/punctuation noise."""
    if s is None:
        return ""
    t = str(s).strip()
    for pat, rep in _LATEX_STRIP:
        t = pat.sub(rep, t)
    t = t.replace("\u2212", "-").replace("\u00b0", "")   # minus sign, degree
    t = t.replace("\u2032", "")                          # prime
    # Strip sentence punctuation from the END only. Stripping it from the FRONT
    # turned ".72" into "72" and ".5" into "5", so a prediction 100x away from
    # the gold was accepted as correct.
    t = re.sub(r"[.\s]+$", "", t.strip())
    t = _UNITS.sub("", t)
    t = re.sub(r"\s+", " ", t).strip().lower()
    t = re.sub(r"[.,;:\s]+$", "", t)
    return t


def _strip_number_decorations(t: str) -> tuple[str, bool]:
    """Remove $, %, thousands commas. Returns (clean, was_percent)."""
    pct = t.endswith("%")
    t = t.rstrip("%").strip()
    t = t.lstrip("$").strip()
    # thousands separators only when they group 3 digits
    if re.fullmatch(r"-?\d{1,3}(,\d{3})+(\.\d+)?", t):
        t = t.replace(",", "")
    return t, pct


# ------------------------------------------------------- safe numeric parse
_LATEX_TO_PY = [
    # A MIXED number a\frac{b}{c} means a + b/c, not a * (b/c). This must run
    # before the plain frac rule, or implicit multiplication reads the real gold
    # "4 \frac { 4 } { 11 }" as 4*(4/11) = 1.4545 instead of 4.3636.
    (re.compile(r"(?<![\d.])(\d+)\s*\\[dt]?frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}"),
     r"((\1)+(\2)/(\3))"),
    # \frac{a}{b} and \dfrac{a}{b}
    (re.compile(r"\\[dt]?frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}"), r"(\1)/(\2)"),
    # \sqrt[n]{x} -> (x)**(1/n)   and   \sqrt{x} -> sqrt(x)
    (re.compile(r"\\sqrt\s*\[([^\]]+)\]\s*\{([^{}]+)\}"), r"(\2)**(1/(\1))"),
    (re.compile(r"\\sqrt\s*\{([^{}]+)\}"), r"sqrt(\1)"),
    (re.compile(r"\\sqrt\s*(\d+)"), r"sqrt(\1)"),
    (re.compile(r"\\pi\b"), "pi"),
    (re.compile(r"\\cdot|\\times"), "*"),
    (re.compile(r"\\div"), "/"),
]

_ALLOWED_NAMES = {"sqrt": math.sqrt, "pi": math.pi, "e": math.e}

# implicit multiplication: 2sqrt(3), 3pi, 2(3+1)
_IMPLICIT = [
    (re.compile(r"(\d)\s*(sqrt|pi|e)\b"), r"\1*\2"),
    (re.compile(r"(\d)\s*\("), r"\1*("),
    (re.compile(r"\)\s*\("), ")*("),
    (re.compile(r"(pi)\s*\("), r"\1*("),
]


def _safe_eval(expr: str) -> float | None:
    """Evaluate a restricted arithmetic expression. Never executes user code."""
    try:
        node = ast.parse(expr, mode="eval")
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        return None

    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant):
            if isinstance(n.value, bool) or not isinstance(n.value, (int, float)):
                raise ValueError("non-numeric constant")
            return float(n.value)
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.UAdd, ast.USub)):
            v = ev(n.operand)
            return v if isinstance(n.op, ast.UAdd) else -v
        if isinstance(n, ast.BinOp):
            a, b = ev(n.left), ev(n.right)
            if isinstance(n.op, ast.Add):
                return a + b
            if isinstance(n.op, ast.Sub):
                return a - b
            if isinstance(n.op, ast.Mult):
                return a * b
            if isinstance(n.op, ast.Div):
                if abs(b) < 1e-15:
                    raise ValueError("division by zero")
                return a / b
            if isinstance(n.op, ast.Pow):
                if abs(b) > 64 or abs(a) > 1e12:      # no expression bombs
                    raise ValueError("exponent out of range")
                return a ** b
            raise ValueError("operator not allowed")
        if isinstance(n, ast.Name):
            if n.id in _ALLOWED_NAMES and not callable(_ALLOWED_NAMES[n.id]):
                return float(_ALLOWED_NAMES[n.id])
            raise ValueError(f"name not allowed: {n.id}")
        if isinstance(n, ast.Call):
            if not isinstance(n.func, ast.Name) or n.func.id != "sqrt":
                raise ValueError("call not allowed")
            if len(n.args) != 1 or n.keywords:
                raise ValueError("bad sqrt arity")
            v = ev(n.args[0])
            if v < 0:
                raise ValueError("sqrt of negative")
            return math.sqrt(v)
        raise ValueError(f"node not allowed: {type(n).__name__}")

    try:
        v = ev(node)
    except (ValueError, TypeError, OverflowError, ZeroDivisionError,
            RecursionError):
        return None
    if not isinstance(v, float) or math.isnan(v) or math.isinf(v):
        return None
    return v


def to_number(s: str) -> float | None:
    """Parse a normalized answer into a float, or None if it is not numeric.

    Handles plain numbers, percentages, thousands separators, simple fractions
    and short LaTeX (sqrt, frac, pi). Anything else returns None and is then
    compared as a string.
    """
    if s is None:
        return None
    t = normalize_text(s)
    if not t:
        return None
    t, pct = _strip_number_decorations(t)
    if not t:
        return None

    try:
        v = float(t)
        return v / 100.0 if pct else v
    except ValueError:
        pass

    if len(t) > 120:                    # not a short answer; refuse
        return None

    # A bare "a-b" is far more often a RANGE or a date span than a subtraction a
    # model left uncomputed, and evaluating it made every equal-width range
    # compare equal (2010-2015 and 2015-2020 both gave -5). Refuse, per the
    # module's rule of resolving ambiguity toward rejection.
    if re.fullmatch(r"\d+\s*[-\u2013]\s*\d+", t):
        return None
    # Two bare literals with only whitespace between them are separate tokens,
    # not a product. Whitespace is deleted below, which used to glue "1 in 5"
    # into 15 once the unit was stripped.
    if re.search(r"\d\s+[.\d]", t):
        return None

    expr = t
    for pat, rep in _LATEX_TO_PY:
        expr = pat.sub(rep, expr)
    expr = expr.replace("^", "**").replace("{", "(").replace("}", ")")
    expr = re.sub(r"\s+", "", expr)
    for pat, rep in _IMPLICIT:
        expr = pat.sub(rep, expr)

    if not re.fullmatch(r"[0-9+\-*/().sqrtpie]*", expr):
        return None
    if not re.search(r"\d|pi|e", expr):
        return None

    v = _safe_eval(expr)
    if v is None:
        return None
    return v / 100.0 if pct else v


# ------------------------------------------------------------- verification
_YESNO = {"yes": "yes", "true": "yes", "y": "yes",
          "no": "no", "false": "no", "n": "no"}

# Words that mean the span around a matched token is NOT a commitment to it.
# Rule 4 below matches a non-numeric gold as a standalone token, which on prose
# accepted "the answer is not increasing" against gold "increasing".
_HEDGE = re.compile(
    r"\b(not|isn't|is not|aren't|no longer|never|cannot|can't|unclear|"
    r"maybe|perhaps|might|probably|unsure|unknown|rather than|instead of|"
    r"neither|nor|except|unlike|opposite|decreas\w*ly not)\b", re.IGNORECASE)

# A committed categorical answer is short. Prose that happens to contain the
# label is not an answer, so cap how much text rule 4 will accept.
_TOKEN_MATCH_MAX_WORDS = 6


ZERO_ABS_TOL = 1e-6      # absolute window used only when the gold is ~0


def numbers_match(pred: float, gold: float, tol: float) -> bool:
    """Relative tolerance, with a tight absolute window when gold is ~0.

    The near-zero branch used to reuse `tol` as an ABSOLUTE bound, so a ChartQA
    gold of 0 accepted anything in [-0.05, 0.05]. A relative tolerance carries no
    meaning at zero, so the window there is a fixed small epsilon instead.
    """
    if math.isclose(gold, 0.0, abs_tol=1e-12):
        return abs(pred) <= ZERO_ABS_TOL
    return abs(pred - gold) <= tol * abs(gold)


def verify_verbose(prediction: str, gold, dataset: str = "default") -> tuple[bool, str]:
    """Return (correct, rule_that_fired). The rule string makes P2 auditable."""
    tol = TOLERANCE.get(dataset, TOLERANCE["default"])

    golds = gold if isinstance(gold, (list, tuple)) else [gold]
    golds = [g for g in golds if g is not None and str(g).strip() != ""]
    if not golds:
        return False, "no_gold"

    pred_raw = extract_answer(prediction) if prediction else ""
    if not pred_raw.strip():
        return False, "no_answer_extracted"

    p_norm = normalize_text(pred_raw)
    if not p_norm:
        return False, "empty_after_normalization"

    p_num = to_number(pred_raw)

    for g in golds:
        g_norm = normalize_text(g)
        g_num = to_number(g)

        # 1. yes/no is its own equivalence class, checked before anything else
        if g_norm in _YESNO:
            if _YESNO.get(p_norm) == _YESNO[g_norm]:
                return True, "yesno_match"
            continue

        # 2. numeric comparison when BOTH sides parse as numbers
        if g_num is not None and p_num is not None:
            if numbers_match(p_num, g_num, tol):
                return True, f"numeric_match(tol={tol})"
            continue

        # 3. exact match after normalization
        if p_norm == g_norm:
            return True, "exact_match"

        # 4. a gold that is not numeric but whose normalized form appears as a
        #    standalone token in the prediction. Deliberately NOT a substring
        #    test: "12" must not match "512".
        #
        #    Two guards, both resolving ambiguity toward rejection. The span must
        #    be a short committed answer rather than reasoning prose, and it must
        #    carry no negation or hedge, or "the answer is not increasing" scores
        #    correct against gold "increasing".
        if g_num is None and len(g_norm) >= 2:
            if len(p_norm.split()) > _TOKEN_MATCH_MAX_WORDS:
                continue
            if _HEDGE.search(p_norm):
                continue
            if re.search(rf"(?<![\w.]){re.escape(g_norm)}(?![\w.])", p_norm):
                return True, "token_match"

    return False, "no_rule_matched"


def verify(prediction: str, gold, dataset: str = "default") -> bool:
    return verify_verbose(prediction, gold, dataset)[0]


# ------------------------------------------------------------------ selftest
def selftest() -> int:
    ok = fail = 0

    def check(cond, label):
        nonlocal ok, fail
        if cond:
            ok += 1
        else:
            fail += 1
            print(f"FAIL  {label}")

    # ---- extraction
    check(extract_answer(r"blah \boxed{42} done") == "42", "boxed")
    check(extract_answer(r"\boxed{1} then \boxed{2}") == "2", "boxed last wins")
    check(extract_answer(r"\boxed{\frac{1}{2}}") == r"\frac{1}{2}", "boxed nested braces")
    check(extract_answer("Answer: 7") == "7", "answer tag")
    check(extract_answer("<answer>9</answer>") == "9", "xml tag")
    check(extract_answer("reasoning\n\n55") == "55", "last line")
    check(extract_answer("") == "", "empty completion")

    # ---- ChartQA relaxed accuracy
    check(verify(r"\boxed{95}", "95", "chartqa"), "chartqa exact numeric")
    check(verify(r"\boxed{97}", "95", "chartqa"), "chartqa within 5%")
    check(not verify(r"\boxed{110}", "95", "chartqa"), "chartqa outside 5%")
    check(verify(r"\boxed{Yes}", "Yes", "chartqa"), "chartqa yes")
    check(not verify(r"\boxed{No}", "Yes", "chartqa"), "chartqa no != yes")
    check(verify(r"\boxed{1,234}", "1234", "chartqa"), "chartqa thousands sep")
    check(verify(r"\boxed{45%}", "45%", "chartqa"), "chartqa percent both")
    check(verify(r"\boxed{$50}", "50", "chartqa"), "chartqa dollar sign")
    check(verify(r"\boxed{2014}", ["2014"], "chartqa"), "chartqa list gold")

    # ---- Geometry3K
    check(verify(r"\boxed{3}", "3", "geometry3k"), "geo exact")
    check(verify(r"\boxed{2 \sqrt{221}}", r"2 \sqrt { 221 }", "geometry3k"), "geo latex sqrt")
    check(verify(r"\boxed{29.7321}", r"2 \sqrt { 221 }", "geometry3k"), "geo numeric vs latex")
    check(verify(r"\boxed{\frac{1}{2}}", "0.5", "geometry3k"), "geo frac")
    check(not verify(r"\boxed{4}", "3", "geometry3k"), "geo wrong")
    check(verify(r"\boxed{120}", "120", "geometry3k"), "geo angle")

    # ---- the failure modes that would FABRICATE complementarity
    check(not verify("I think the answer might be around 90 or so", "95", "chartqa"),
          "prose without an answer must not pass")
    check(not verify(r"\boxed{}", "5", "chartqa"), "empty box fails")
    check(not verify("", "5", "chartqa"), "empty completion fails")
    check(not verify(r"\boxed{512}", "12", "chartqa"), "substring 12 in 512 must fail")
    check(not verify(r"\boxed{0}", "95", "chartqa"), "zero is not within 5% of 95")
    check(not verify(r"\boxed{95}", "", "chartqa"), "blank gold fails")
    # tolerance must not be so loose that neighbouring golds collide
    check(not verify(r"\boxed{6}", "5", "chartqa"), "6 is not within 5% of 5")

    # ---- the evaluator must never execute anything
    check(to_number("__import__('os').system('echo hi')") is None, "no code execution")
    check(to_number("open('/etc/passwd')") is None, "no builtins")
    check(to_number("9**9**9") is None, "no exponent bomb")
    check(to_number("1/0") is None, "division by zero rejected")
    check(to_number("x + 1") is None, "unknown name rejected")

    # ---- numeric parsing
    check(abs(to_number("50%") - 0.5) < 1e-12, "percent to fraction")
    check(abs(to_number(r"\frac{3}{4}") - 0.75) < 1e-12, "latex frac")
    check(abs(to_number(r"2\sqrt{9}") - 6.0) < 1e-9, "implicit mult sqrt")
    check(abs(to_number(r"3\pi") - 3 * math.pi) < 1e-9, "implicit mult pi")
    check(to_number("hello") is None, "non-numeric text")
    check(to_number("") is None, "empty")

    # ---- zero gold uses an absolute fallback, never a relative one
    check(verify(r"\boxed{0}", "0", "chartqa"), "zero gold exact")
    check(not verify(r"\boxed{1}", "0", "chartqa"), "zero gold rejects 1")

    # ---- v2.0 regression pack -------------------------------------------
    # Every case below is a defect that was live in v1.0 and was confirmed by
    # execution against the stored rollouts before being fixed. They are kept
    # as tests because each one silently corrupted a measurement rather than
    # raising, which is the failure mode this module exists to prevent.

    # Degree marks in LaTeX. The unicode sign and the word were handled, the
    # LaTeX spellings were not. Measured impact: 237 flipped verdicts over
    # 57,600 stored rollouts, concentrated ~20x on the untrained base arm,
    # because this verifier is also the training reward.
    for _p in (r"60^\circ", r"60^{\circ}", r"60\circ", r"60\degree",
               "60 degrees", "60°"):
        check(verify(_p, "60", "geometry3k"), f"degree form {_p!r}")
    check(not verify(r"68^\circ", "47", "geometry3k"), "wrong angle still rejected")

    # A unit used to be stripped without its exponent, so "37 cm^2" became
    # "37 ^2" and evaluated to 1369.
    check(verify(r"37 cm^2", "37", "geometry3k"), "squared unit")
    check(not verify(r"37 cm^2", "1369", "geometry3k"), "squared unit not exponent")

    # A mixed number is a sum, not a product. The real gold "4 \frac{4}{11}"
    # parsed as 4*(4/11) = 1.4545 instead of 4.3636.
    check(abs(to_number(r"4\frac{4}{11}") - 4.363636) < 1e-4, "mixed number")
    check(abs(to_number(r"3\frac{1}{2}") - 3.5) < 1e-9, "mixed number half")
    check(abs(to_number(r"\frac{3}{4}") - 0.75) < 1e-9, "plain frac unchanged")

    # Leading punctuation was stripped, turning ".72" into 72 and accepting a
    # prediction 100x from the gold. FALSE POSITIVE: the dangerous direction.
    check(not verify(".72", "72", "geometry3k"), "leading decimal not stripped")
    check(not verify(".5", "5", "geometry3k"), "leading decimal 10x")
    check(verify(".5", "0.5", "geometry3k"), "leading decimal still parses")

    # A bare "a-b" is a range far more often than an uncomputed subtraction,
    # and evaluating it made every equal-width range compare equal.
    check(to_number("2010-2015") is None, "range not subtracted")
    check(not verify("2010-2015", "2015-2020", "chartqa"), "ranges not equal")
    check(abs(to_number("-5") + 5) < 1e-9, "negative still parses")

    # Whitespace deletion glued separate tokens into one number.
    check(to_number("1 in 5") is None, "adjacent literals refused")

    # A trailing empty box used to win last-occurrence and erase the answer.
    check(extract_answer(r"a \boxed{60} b \boxed{}") == "60", "last non-empty box")
    check(extract_answer(r"\boxed{}") == "", "all-empty box stays empty")

    # token_match accepted prose that merely contained the label, including
    # outright negations. FALSE POSITIVE.
    check(not verify("The answer is not increasing", "increasing", "chartqa"),
          "negation rejected")
    check(verify("increasing", "increasing", "chartqa"), "bare label still ok")

    # A relative tolerance is meaningless at zero; it used to be reused as an
    # absolute window, so a gold of 0 accepted anything within 0.05.
    check(not verify("0.04", "0", "chartqa"), "near-zero window tight")
    check(verify("0", "0", "chartqa"), "zero still matches zero")

    # Presentation-only decoration must not block a numeric match.
    for _p in (r"\mathbf{60}", r"\textbf{60}", r"60\quad", r"\hspace{2pt}60",
               r"\approx 60", r"60^\prime", r"\overline{60}"):
        check(verify(_p, "60", "geometry3k"), f"decoration {_p!r}")
    check(verify(r"\$50", "50", "chartqa"), "escaped dollar")

    print(f"csm_verify selftest: {ok} passed, {fail} failed")
    if fail:
        print("\nA FAILING VERIFIER INVALIDATES EVERY MEASUREMENT DOWNSTREAM.")
        return 1
    print("Guards fire: no code execution, no substring leakage, no empty-answer pass.")
    print("This is not a substitute for a hand audit of real decisions; "
          "it only proves the rules behave as written.")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "selftest":
        sys.exit(selftest())
    print(__doc__)
    print("usage: python csm_verify.py selftest")
