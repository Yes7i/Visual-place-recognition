#!/usr/bin/env bash
# 打开视觉位置识别网页:启动本地 Web 服务并自动打开浏览器
# 用法(在项目目录): ./open_web.sh
cd "$(dirname "$0")"
exec python3 webui.py
