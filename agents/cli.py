"""
TechNews Intelligence - Agent CLI
==================================
本地终端交互入口，用于测试和使用 Agent。
使用方法：
    1. 将/agents目录下 .env.example 复制为 .env 并填入真实配置
    2. pip install -r requirements.txt
    3. python cli.py
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv


def main():
    # 加载 .env（优先查找同目录下的 .env）
    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.exists():
        print(f"[错误] 找不到 .env 文件: {env_path}")
        sys.exit(1)
    load_dotenv(dotenv_path=env_path, override=True)

    # 设置网络代理（可选）
    http_proxy = os.getenv("HTTP_PROXY", "")
    https_proxy = os.getenv("HTTPS_PROXY", "")
    if http_proxy:
        os.environ["HTTP_PROXY"] = http_proxy
    if https_proxy:
        os.environ["HTTPS_PROXY"] = https_proxy

    # 延迟导入，确保环境变量已经加载
    from core import create_agent_chat, close_db_pool

    print("正在初始化 Agent")

    try:
        chat = create_agent_chat()
        print("初始化完成！输入 exit 退出。")
        print("示例: '英伟达最新GPU发布有什么影响'")
        print("支持追问: '那它的竞争对手呢'、'再深入说说第二点'\n")

        while True:
            try:
                user_input = input("你: ").strip()
                if not user_input:
                    continue
                if user_input.lower() in ["exit", "quit"]:
                    break

                response = chat.send_message(user_input)

                if getattr(response, "candidates", None) is None or not response.candidates:
                    print("Agent: [错误] 模型未能生成任何候选内容，可能触发了平台安全性拦截机制。\n")
                    continue

                parts_texts = [
                    part.text
                    for part in response.candidates[0].content.parts
                    if hasattr(part, "text") and part.text
                ]
                response_text = "".join(parts_texts) if parts_texts else (
                    "[错误] 模型未能返回有效文本，可能触发了安全拦截或遇到了解析异常。"
                )
                print(f"Agent: {response_text}\n")

            except KeyboardInterrupt:
                print("\n[提示] 操作已手动中断（Ctrl+C）。可以继续提问，或输入 exit 退出。")
            except Exception as e:
                print(f"[错误] {e}\n")

    except Exception as e:
        print(f"[错误] 初始化失败: {e}")
        print("请检查 .env 配置或网络连接。")
    finally:
        close_db_pool()
        print("已退出，数据库连接已关闭。")


if __name__ == "__main__":
    main()
