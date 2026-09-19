"""单元测试：重点覆盖"模型输出不规范"这种情况。

为什么专门测这个？因为 LLM 的输出是不可信的输入源。
它偶尔会包 ```json 代码块、夹带解释文字、少字段、编动作名。
这些必须在解析层就被挡住，而不是让脏数据流进主循环。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bagent.models import Usage, parse_action  # noqa: E402


def test_parse_plain_json():
    action, err = parse_action('{"thought": "点下一页", "action": "click", "ref": 2}')
    assert err == ""
    assert action is not None
    assert action.action == "click"
    assert action.ref == 2


def test_parse_json_in_markdown_fence():
    raw = '```json\n{"thought": "x", "action": "scroll", "dy": 600}\n```'
    action, err = parse_action(raw)
    assert err == ""
    assert action is not None and action.dy == 600


def test_parse_json_with_surrounding_prose():
    raw = '好的，我决定这样做：\n{"action": "finish", "answer": "作者是爱因斯坦"}\n以上。'
    action, err = parse_action(raw)
    assert err == ""
    assert action is not None and action.answer == "作者是爱因斯坦"


def test_reject_unknown_action_name():
    action, err = parse_action('{"action": "launch_missile"}')
    assert action is None
    assert "不合法" in err or "validation" in err.lower()


def test_reject_missing_action_field():
    action, err = parse_action('{"thought": "我在想"}')
    assert action is None


def test_reject_empty_and_garbage():
    for raw in ("", "   ", "我不知道该怎么办"):
        action, err = parse_action(raw)
        assert action is None
        assert err


def test_usage_accumulates_and_costs():
    u = Usage()
    u.add(Usage(prompt_tokens=1000, completion_tokens=500, calls=1))
    u.add(Usage(prompt_tokens=2000, completion_tokens=1000, calls=1))
    assert u.prompt_tokens == 3000
    assert u.completion_tokens == 1500
    assert u.calls == 2
    assert u.total_tokens == 4500
    # 单价 1 元/百万输入、2 元/百万输出
    assert u.cost_yuan(1.0, 2.0) == pytest.approx(3000 / 1e6 * 1.0 + 1500 / 1e6 * 2.0)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
