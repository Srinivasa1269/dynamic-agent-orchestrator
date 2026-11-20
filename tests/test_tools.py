"""
test_tools.py - Checks the calculator and web search tools.

The web search tests swap the real search for a fake one, so they need no
internet. One "live" test at the bottom does a real search.
"""

import pytest

from app import tools
from app.tools import CalculatorError, calculate, calculator, web_search


# --- Calculator -------------------------------------------------------------
@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("2 + 3 * 4", 14),
        ("(2 + 3) * 4", 20),
        ("10 / 4", 2.5),
        ("7 // 2", 3),
        ("7 % 2", 1),
        ("2 ** 10", 1024),
        ("-5 + 2", -3),
        ("sqrt(16)", 4.0),
        ("round(10 / 3, 2)", 3.33),
        ("max(1, 5, 3)", 5),
    ],
)
def test_calculator_does_maths(expression: str, expected: float) -> None:
    assert calculate(expression) == expected


def test_calculator_knows_pi() -> None:
    assert calculate("round(pi, 2)") == 3.14


@pytest.mark.parametrize(
    "dangerous",
    [
        "__import__('os').system('echo hacked')",
        "open('secret.txt').read()",
        "().__class__.__bases__",
        "[1, 2, 3]",
        "'a' * 5",
        "x + 1",
        "print(1)",
        "1 if True else 2",
        "lambda: 1",
        "9 ** 9 ** 9 ** 9",
        "2 ** 5000",
        "1 / 0",
        "",
        "2 +",
    ],
)
def test_calculator_refuses_anything_that_is_not_plain_maths(dangerous: str) -> None:
    with pytest.raises(CalculatorError):
        calculate(dangerous)


def test_calculator_tool_returns_text() -> None:
    assert calculator.invoke({"expression": "6 * 7"}) == "42"


def test_calculator_tool_reports_errors_as_text_not_crashes() -> None:
    answer = calculator.invoke({"expression": "__import__('os')"})
    assert answer.startswith("Error:")


# --- Web search -------------------------------------------------------------
def test_web_search_formats_results(monkeypatch) -> None:
    fake_results = [
        {"title": "First", "url": "https://example.com/1", "snippet": "One."},
        {"title": "Second", "url": "https://example.com/2", "snippet": "Two."},
    ]
    monkeypatch.setattr(tools, "search_web", lambda query: fake_results)
    answer = web_search.invoke({"query": "anything"})
    assert "1. First" in answer
    assert "https://example.com/2" in answer


def test_web_search_handles_no_results(monkeypatch) -> None:
    monkeypatch.setattr(tools, "search_web", lambda query: [])
    assert "No results" in web_search.invoke({"query": "anything"})


def test_web_search_reports_failures_as_text_not_crashes(monkeypatch) -> None:
    def broken(query: str):
        raise ConnectionError("no internet")

    monkeypatch.setattr(tools, "search_web", broken)
    assert web_search.invoke({"query": "anything"}).startswith("Error:")


def test_search_library_no_results_error_means_an_empty_list(monkeypatch) -> None:
    # The search library raises an error when it finds nothing.
    import ddgs

    class FakeDDGS:
        def text(self, query, **kwargs):
            raise RuntimeError("No results found.")

    monkeypatch.setattr(ddgs, "DDGS", FakeDDGS)
    assert tools.search_web("anything") == []


def test_search_skips_adverts(monkeypatch) -> None:
    import ddgs

    class FakeDDGS:
        def text(self, query, **kwargs):
            return [
                {"title": "Ad", "href": "https://www.bing.com/aclick?ld=abc", "body": "Buy!"},
                {"title": "Real", "href": "https://example.com/page", "body": "Facts."},
            ]

    monkeypatch.setattr(ddgs, "DDGS", FakeDDGS)
    assert [r["title"] for r in tools.search_web("anything")] == ["Real"]


def test_web_search_rejects_empty_query() -> None:
    assert web_search.invoke({"query": "   "}).startswith("Error:")


@pytest.mark.live
def test_live_web_search_returns_results() -> None:
    results = tools.search_web("python programming language", max_results=2)
    assert results and results[0]["url"].startswith("http")
