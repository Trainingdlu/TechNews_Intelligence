"""Local CLI entrypoint for TechNews agent."""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv


def _extract_response_text(response) -> str:
    text = getattr(response, "text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()

    if isinstance(response, str):
        return response.strip()

    return "[Error] Agent returned no valid text."


def main():
    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.exists():
        print(f"[Error] .env not found: {env_path}")
        sys.exit(1)
    load_dotenv(dotenv_path=env_path, override=True)

    http_proxy = os.getenv("HTTP_PROXY", "")
    https_proxy = os.getenv("HTTPS_PROXY", "")
    if http_proxy:
        os.environ["HTTP_PROXY"] = http_proxy
    if https_proxy:
        os.environ["HTTPS_PROXY"] = https_proxy

    from agent import create_agent_chat
    from db import close_db_pool

    print("Initializing agent")

    try:
        chat = create_agent_chat()
        print("Ready. Type 'exit' to quit.")

        while True:
            try:
                user_input = input("You: ").strip()
                if not user_input:
                    continue
                if user_input.lower() in ["exit", "quit"]:
                    break

                response = chat.send_message(user_input)
                response_text = _extract_response_text(response)
                print(f"Agent: {response_text}\n")

            except KeyboardInterrupt:
                print("\n[Info] Interrupted. Continue typing or use 'exit' to quit.")
            except Exception as e:
                print(f"[Error] {e}\n")

    except Exception as e:
        print(f"[Error] Initialization failed: {e}")
        print("Please verify .env and network settings.")
    finally:
        close_db_pool()
        print("Exited. Database connections closed.")


if __name__ == "__main__":
    main()
