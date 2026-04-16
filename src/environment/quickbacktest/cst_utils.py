from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import libcst as cst
from libcst import matchers as m


@dataclass(frozen=True)
class JsonExtraction:
    json_text: str
    prefix: str
    suffix: str


def extract_first_json_object(text: str) -> Optional[JsonExtraction]:
    """
    从任意文本中提取第一个完整 JSON object（{...}），并返回：
    - json_text: JSON 原文
    - prefix: JSON 前的文本（原样保留）
    - suffix: JSON 后的文本（原样保留）

    处理要点：
    - 支持嵌套 {}
    - 正确处理字符串字面量中的 '{' '}'（不计入括号层级）
    - 正确处理转义 \" \\ 等
    """
    s = text
    n = len(s)
    i = 0

    # 找到第一个 '{'
    while i < n and s[i] != "{":
        i += 1
    if i >= n:
        return None

    start = i
    depth = 0
    in_str = False
    esc = False

    while i < n:
        ch = s[i]

        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    return JsonExtraction(
                        json_text=s[start:end],
                        prefix=s[:start],
                        suffix=s[end:],
                    )

        i += 1

    return None  # 没找到闭合


def parse_json_docstring(ss: cst.SimpleString) -> Optional[Tuple[Dict[str, Any], JsonExtraction]]:
    """
    从 docstring 中解析 JSON block（允许前后混杂文字）。
    返回 (payload, extraction)，其中 extraction 包含 prefix/suffix 用于回填保留。
    """
    text = ss.evaluated_value if isinstance(ss.evaluated_value, str) else ""
    ex = extract_first_json_object(text)
    if not ex:
        return None
    try:
        return json.loads(ex.json_text), ex
    except json.JSONDecodeError:
        raise json.JSONDecodeError


def to_docstring_from_text(text: str) -> cst.SimpleString:
    """
    把“完整 docstring 内容文本”包装成三引号 SimpleString。
    注意：这里统一使用 \"\"\"...\"\"\"；如果你想保留原引号风格，后面我也给你方案。
    """
    # 避免最外层直接紧贴 """，更符合常见风格
    return cst.SimpleString(f'"""{text}"""')


# def add_only(base: Dict[str, Any], add: Dict[str, Any]) -> Dict[str, Any]:
#     """
#     深度“只补缺失”：
#     - base 没有的 key 才补
#     - 如果 base[k] 和 add[k] 都是 dict，则递归补缺失
#     - 其它类型：base 已有就不覆盖
#     """
#     out = dict(base)
#     for k, v in add.items():
#         if k not in out:
#             out[k] = v

#             if isinstance(out[k], dict) and isinstance(v, dict):
#                 out[k] = add_only(out[k], v)
#     return out


def update_dict_only(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """
    深度“只更新已有”：
    - base 有的 key 才更新
    - 如果 base[k] 和 updates[k] 都是 dict，则递归更新
    - 其它类型：base 已有就覆盖
    """
    out = dict(base)
    for k, v in updates.items():
        if k in out:
            if isinstance(out[k], dict) and isinstance(v, dict):
                out[k] = update_dict_only(out[k], v)
            else:
                out[k] = v
    return out

@dataclass(frozen=True)
class PatchConfig:
    add_fields: Dict[str, Any]


class Patch(cst.CSTTransformer):
    def __init__(self, config: Optional[PatchConfig] = None) -> None:
        super().__init__()
        if config is None:
            config = PatchConfig(add_fields={"version": "1.0"})
        self.config = config

    def leave_ClassDef(self, original: cst.ClassDef, updated: cst.ClassDef) -> cst.ClassDef:
        body = list(updated.body.body)
        if not body:
            return updated

        # docstring 语句：第一句是 Expr(SimpleString)
        first = body[0]
        if not m.matches(first, m.SimpleStatementLine(body=[m.Expr(value=m.SimpleString())])):
            return updated

        stmt = first  # SimpleStatementLine
        ss = stmt.body[0].value  # SimpleString

        parsed = parse_json_docstring(ss)
        if parsed is None:
            return updated  # 非 JSON docstring：不动

        payload, ex = parsed

        # ===== 增量规则 =====
        payload = update_dict_only(payload, self.config.add_fields)
        # ====================

        # 只替换 JSON block，其它文字原样保留
        new_json = json.dumps(payload, ensure_ascii=True, indent=3)
        new_text = f"{ex.prefix}{new_json}{ex.suffix}"

        new_stmt = stmt.with_changes(body=[cst.Expr(value=to_docstring_from_text(new_text))])
        body[0] = new_stmt
        return updated.with_changes(body=updated.body.with_changes(body=body))


def patch_file(path: str, config: Optional[PatchConfig] = None) -> None:
    p = Path(path)
    code = p.read_text()
    mod = cst.parse_module(code)
    new = mod.visit(Patch(config=config))
    if new.code != code:
        p.write_text(new.code)

