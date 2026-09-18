"""第 2 章快速上手实验：第一个 Deep Agent（Hello World + 自定义工具）

课程：https://datawhalechina.github.io/deepagents-in-action/chapters/ch02-quickstart/
环境：复用 research_deepagent 的 .venv（deepagents 0.7.13 已装）
运行：source ~/research_deepagent/.env 不需要；SILICONFLOW_API_KEY/MODEL_NAME 在 ~/.zshrc
"""

import os

from langchain_openai import ChatOpenAI

from deepagents import create_deep_agent

# 通过硅基流动接入模型（OpenAI 兼容接口）
model = ChatOpenAI(
    model=os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct"),
    api_key=os.environ["SILICONFLOW_API_KEY"],
    base_url="https://api.siliconflow.cn/v1",
)


# ---------- 实验 1：Hello World ----------
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"


agent = create_deep_agent(
    model=model,
    tools=[get_weather],
    system_prompt="You are a helpful assistant.",
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "北京今天天气怎么样？"}]}
)
print("=== 实验 1：Hello World ===")
print(result["messages"][-1].content)
print()


# ---------- 实验 2：计算器 Agent（工具三要素练习） ----------
def calculate(expression: str) -> str:
    """Evaluate a math expression and return the result.

    Args:
        expression: A pure math expression with numbers only, e.g. "1 + 2 * 3".
            Variable names or Python code are NOT allowed.
    """
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))  # noqa: S307 仅演示用
    except Exception as e:  # noqa: BLE001
        return f"Invalid math expression: {expression!r} ({type(e).__name__}). 只接受纯数字算式，不接受变量名或代码。"


def convert_currency(
    amount: float, from_currency: str, to_currency: str = "CNY"
) -> dict:
    """Convert an amount from one currency to another.

    Args:
        amount: The amount to convert.
        from_currency: The source currency code, e.g. "USD".
        to_currency: The target currency code, defaults to "CNY".
    """
    rates = {"USD": 7.2, "CNY": 1.0, "EUR": 7.8}
    cny = amount * rates[from_currency]
    return {"amount": round(cny / rates[to_currency], 2), "currency": to_currency}


calc_agent = create_deep_agent(
    model=model,
    tools=[calculate, convert_currency],
    system_prompt="你是一个计算助手，能帮用户做数学运算和货币换算。" "多步任务请分步调用工具，每步工具调用只传纯数字算式，不要传变量名。",
)

result = calc_agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "帮我把 100 美元换算成人民币，再用它乘以 1.08 的通胀系数。",
            }
        ]
    }
)
print("=== 实验 2：计算器 Agent ===")
print(result["messages"][-1].content)
