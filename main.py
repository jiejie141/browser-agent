"""项目入口。

VSCode 里直接按 F5 就会跑这个文件。
它做的事很简单：把 src 加进模块搜索路径，然后把控制权交给 cli。

为什么需要这一行？因为 Python 默认只认"当前目录"和"已安装的包"，
src/ 这种布局必须显式告诉解释器去哪儿找 bagent 这个包。
（专业做法是写 pyproject.toml 用 pip install -e . 装成可编辑包，
  但对新手来说多一步；等你熟了再换。）
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bagent.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
