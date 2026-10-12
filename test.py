import sys

from ollama import ResponseError
from langchain_ollama import ChatOllama


def main() -> int:
    model_name = "qwen2.5:0.5b"
    model = ChatOllama(
        model=model_name,
        num_ctx=512,
        num_predict=128,
    )

    while True:
        try:
            prompt = input("ask (or 'quit' to exit): ").strip()
        except EOFError:
            print()
            return 0

        if prompt.lower() in {"quit", "exit"}:
            return 0
        if not prompt:
            print("Please enter a prompt.", file=sys.stderr)
            continue

        try:
            response = model.invoke(prompt)
        except ResponseError as exc:
            error = str(exc)
            if exc.status_code == 404:
                print(
                    f"Model {model_name!r} is not installed. "
                    f"Run `ollama pull {model_name}` and try again.",
                    file=sys.stderr,
                )
            elif "allocate" in error.lower() or "memory" in error.lower():
                print(
                    "Ollama could not allocate enough memory for this model. "
                    "Close other applications and try again.",
                    file=sys.stderr,
                )
            else:
                print(f"Ollama request failed: {error}", file=sys.stderr)
            continue

        print(response.content)


if __name__ == "__main__":
    raise SystemExit(main())
