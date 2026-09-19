# browser-agent 容器镜像。
#
# 和 pr-review-agent 的镜像有个关键差别：这个项目**需要浏览器**。
# Playwright 的 Chromium 体积不小（~180MB），而且它对系统库有依赖，
# 所以不能像纯 stdlib 项目那样只 COPY 源码就完事。
#
# 这里直接用 Playwright 官方镜像：它已经装好了 Chromium 和全部系统依赖，
# 自己从头 apt install 一遍既慢又容易漏库（经典坑：缺 libnss3 / libatk，
# 报错信息还很不直观）。
FROM mcr.microsoft.com/playwright/python:v1.49.0-jammy

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src

WORKDIR /app

# 先装依赖，利用 Docker 层缓存：只改源码时不会重装 playwright
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 官方镜像已带浏览器，但版本要和 pip 装的 playwright 对齐，装一次更稳妥
RUN playwright install --with-deps chromium

COPY src/ ./src/
COPY main.py ./
COPY docs/ ./docs/
COPY tasks/ ./tasks/

# 非 root 运行。浏览器里跑的是「模型生成的页面操作」，权限必须收窄。
RUN useradd --create-home --shell /bin/bash app \
    && mkdir -p /app/runs /app/.cache \
    && chown -R app:app /app
USER app

# 让 Playwright 把浏览器缓存指向 app 用户可写的位置
ENV PLAYWRIGHT_BROWSERS_PATH=/ms-playwright

# 容器内默认无头 + 离线替身：CI 里不联网、不烧 token、结果可复现
ENV HEADLESS=true \
    MOCK=true \
    OFFLINE=true

ENTRYPOINT ["python", "main.py"]
CMD ["--help"]
