from __future__ import annotations

import re
from warnings import warn
from pathlib import Path
from typing import Any
import pkgutil
from pprint import pprint
from importlib import import_module
import inspect
import importlib


MOD = None
MOD_NAME = "sensitivity_jax"
DELIM = "."


def get_summary_line(docs):
    try:
        line = docs.split("\n\n")[0].replace("\n", " ")
        line = re.sub(r"\s+", r" ", line)
        return line
    except:
        return ""


def find_all_modules(root_mod, modules_dir=None, prefix=None):
    modules_dir = dict() if modules_dir is None else modules_dir
    for _, module_name, ispkg in pkgutil.iter_modules(root_mod.__path__):
        module = importlib.import_module(f"{root_mod.__name__}.{module_name}")
        function_list = []
        for name, obj in inspect.getmembers(module):
            if inspect.isfunction(obj) and obj.__doc__ is not None and name[:1] != "_":
                if obj.__module__ != module.__name__:
                    continue
                summary = get_summary_line(obj.__doc__)
                sig = inspect.signature(obj)
                full_name = f"{prefix}{DELIM}{name}" if prefix is not None else name
                function_list.append(
                    dict(name=full_name, summary=summary, signature=sig, doc=obj.__doc__)
                )
        full_name = f"{prefix}{DELIM}{module_name}" if prefix is not None else module_name
        if ispkg:
            find_all_modules(module, modules_dir, prefix=full_name)
        modules_dir[full_name] = function_list
    return modules_dir


def make_table(table_list: list[list[Any]]) -> str:
    assert len(table_list)
    n_columns = len(table_list[0])
    assert all(len(row) == n_columns for row in table_list)
    column_lengths = [0 for _ in range(n_columns)]
    for row in table_list:
        for i, el in enumerate(row):
            column_lengths[i] = max(column_lengths[i], len(f" {el} "))
    # make table
    body = ""
    for j, row in enumerate(table_list):
        body += (
            "|"
            + "|".join([f" {el:^{column_lengths[i] - 2}} " for (i, el) in enumerate(row)])
            + "|\n"
        )
        if j == 0:
            body += "|" + "|".join(["-" * column_len for column_len in column_lengths]) + "|\n"
    return body


def make_the_page(
    fname: str,
    name: str,
    signature: inspect.Signature,
    doc: str,
    path: Path | str,
    prev_page: str | None = None,
    next_page: str | None = None,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = f"""
#

::: {fname}{DELIM}{name.split(DELIM)[-1]}
"""
    if prev_page is not None or next_page is not None:
        content += "\n<div class='container'>\n"
    if prev_page is not None:
        content += f"<div class='left-div'><a href='{prev_page[5]}'><<< prev<p>{prev_page[1]}</p></a></div>"
    if next_page is not None:
        content += f"<div class='right-div'><a href='{next_page[5]}'>next >>><p>{next_page[1]}</p></a></div>"
    if prev_page is not None or next_page is not None:
        content += "</div>"
    path.write_text(content)


def main():
    global MOD, MOD_NAME
    overview_file = Path("docs/api/overview.md")
    if overview_file.exists():
        warn(
            "Overview file exists, we will not rebuild the documentation. "
            + "Please delete the `docs/api` folder if you want to force a rebuild"
        )
        return
    if MOD is None:
        MOD = import_module(MOD_NAME)
    overview_page = ""
    modules_dir = find_all_modules(MOD, prefix=MOD.__name__)
    pages = []
    for fname, fn_list in modules_dir.items():
        table_rows = [["name", "summary"]]
        if len(fn_list) > 0:
            for fn in fn_list:
                arg_list = ", ".join(list(fn["signature"].parameters.keys()))
                name: str = fn["name"]
                summary = fn["summary"]
                path = f"./docs/api/{fname.replace(DELIM, '/')}/{name.split(DELIM)[-1]}.md"
                url = f"/{MOD.__name__}/api" / Path(path).relative_to(Path(*Path(path).parts[:2]))
                url = url.with_suffix("")
                table_rows.append([f"[{name.split(DELIM)[-1]}({arg_list})]({url})", summary])
                pages.append((fname, name, fn["signature"], fn["doc"], path, url))
            overview_page += f"\n\n# `{DELIM.join(fname.split(DELIM)[1:])}`\n\n"
            overview_page += make_table(table_rows)
    overview_file.parent.mkdir(parents=True, exist_ok=True)
    overview_file.write_text(overview_page)
    for i, (fname, name, signature, doc, path, url) in enumerate(pages):
        prev_page = pages[i - 1] if i > 0 else None
        next_page = pages[i + 1] if i < len(pages) - 1 else None
        make_the_page(fname, name, signature, doc, path, prev_page, next_page)
    pprint([page[0] for page in pages])


####################################################################################################


def on_pre_build(*args, **kwargs):
    main()


if __name__ == "__main__":
    main()
