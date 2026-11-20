"""
tools.py - Web search and safe calculator tools for workers. (Phase 3)

What this file does, in simple words:
- A "tool" is a function the AI model is allowed to call while it works.
- web_search looks things up on the internet (using DuckDuckGo, no key needed).
- calculator does maths exactly, so the model does not have to guess numbers.

Safety:
- The calculator NEVER uses Python's eval(). eval() would run any code the model
  typed, which is dangerous. Instead we read the maths expression piece by
  piece and allow only numbers, + - * / and a short list of maths functions.
- Tools never crash the worker. When something goes wrong they return a
  message starting with "Error:", so the model can read it and try again.
"""

import ast
import math
import operator
from typing import Any

from langchain_core.tools import tool

from app import config

# --- Calculator -------------------------------------------------------------
# The only operations the calculator understands.
_BINARY_OPERATIONS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPERATIONS = {ast.UAdd: operator.pos, ast.USub: operator.neg}

# The only functions and named numbers the calculator understands.
_FUNCTIONS = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sqrt": math.sqrt,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "floor": math.floor,
    "ceil": math.ceil,
}
_CONSTANTS = {"pi": math.pi, "e": math.e}

# Guards against expressions built to freeze the computer, like 9**9**9**9.
_MAX_EXPRESSION_LENGTH = 300
_MAX_EXPONENT = 1000
_MAX_NUMBER = 1e100


class CalculatorError(Exception):
    """Raised when an expression is not allowed or cannot be worked out."""


def _evaluate(node: ast.AST) -> Any:
    """Work out one piece of the expression. Anything not listed is refused."""
    if isinstance(node, ast.Constant):
        # bool is a kind of int in Python, so we rule it out explicitly.
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        raise CalculatorError("only numbers are allowed")

    if isinstance(node, ast.Name):
        if node.id in _CONSTANTS:
            return _CONSTANTS[node.id]
        raise CalculatorError(f"unknown name '{node.id}'")

    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATIONS:
        return _UNARY_OPERATIONS[type(node.op)](_evaluate(node.operand))

    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATIONS:
        left = _evaluate(node.left)
        right = _evaluate(node.right)
        if isinstance(node.op, ast.Pow) and (
            abs(right) > _MAX_EXPONENT or abs(left) > _MAX_NUMBER
        ):
            raise CalculatorError("the power is too large")
        result = _BINARY_OPERATIONS[type(node.op)](left, right)
        if isinstance(result, (int, float)) and abs(result) > _MAX_NUMBER:
            raise CalculatorError("the result is too large")
        return result

    if isinstance(node, ast.Call):
        if (
            isinstance(node.func, ast.Name)
            and node.func.id in _FUNCTIONS
            and not node.keywords
        ):
            arguments = [_evaluate(argument) for argument in node.args]
            return _FUNCTIONS[node.func.id](*arguments)
        raise CalculatorError("that function is not allowed")

    raise CalculatorError("only simple maths is allowed")


def calculate(expression: str) -> float | int:
    """Safely work out a maths expression such as "(1200 * 0.15) / 3".

    Raises CalculatorError if the expression is not plain maths.
    """
    expression = expression.strip()
    if not expression:
        raise CalculatorError("the expression is empty")
    if len(expression) > _MAX_EXPRESSION_LENGTH:
        raise CalculatorError("the expression is too long")

    try:
        tree = ast.parse(expression, mode="eval")
        result = _evaluate(tree.body)
    except CalculatorError:
        raise
    except ZeroDivisionError:
        raise CalculatorError("division by zero") from None
    except (SyntaxError, ValueError, TypeError, OverflowError, RecursionError) as error:
        raise CalculatorError(f"could not work this out ({type(error).__name__})") from None

    if isinstance(result, complex):
        raise CalculatorError("the result is not a real number")
    return result


@tool
def calculator(expression: str) -> str:
    """Work out a maths expression exactly and return the result.

    Use this for any arithmetic instead of working it out yourself.
    Supports + - * / // % ** and brackets, the functions abs, round, min, max,
    sqrt, log, log10, exp, sin, cos, tan, floor, ceil, and the constants pi
    and e. Example: "round((1200 * 0.15) / 3, 2)".
    """
    try:
        return str(calculate(expression))
    except CalculatorError as error:
        return f"Error: {error}. Send plain maths only, for example: (2 + 3) * 4"


# --- Web search -------------------------------------------------------------
def search_web(query: str, max_results: int | None = None) -> list[dict[str, str]]:
    """Search the web and return a list of {"title", "url", "snippet"} results."""
    # Imported here so the rest of the project loads even without internet.
    from ddgs import DDGS

    limit = max_results or config.SEARCH_MAX_RESULTS
    try:
        raw_results = DDGS().text(query, max_results=limit)
    except Exception as error:
        # The search library raises an error when a search simply finds
        # nothing. That is not a failure, so we return an empty list.
        if "no results" in str(error).lower():
            return []
        raise
    return [
        {
            "title": str(item.get("title", "")),
            "url": str(item.get("href", "")),
            "snippet": str(item.get("body", "")),
        }
        for item in raw_results
        # Skip paid adverts, which arrive as tracking links, not real pages.
        if "/aclick?" not in str(item.get("href", ""))
    ]


@tool
def web_search(query: str) -> str:
    """Search the web and return the top results (title, URL and a short snippet).

    Use this to find facts, figures and recent information. Write the query
    the way you would type it into a search engine. Search again with
    different words if the first results are not useful.
    """
    query = query.strip()
    if not query:
        return "Error: the search query is empty."

    try:
        results = search_web(query)
    except Exception as error:  # no internet, search service busy, ...
        return f"Error: the search failed ({type(error).__name__}). Try again or rephrase."

    if not results:
        return "No results found. Try different search words."

    lines = []
    for number, item in enumerate(results, start=1):
        lines.append(f"{number}. {item['title']}\n   {item['url']}\n   {item['snippet']}")
    return "\n\n".join(lines)


# The full toolbox every worker receives.
WORKER_TOOLS = [web_search, calculator]
