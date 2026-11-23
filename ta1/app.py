"""
app.py
------
Entry point for the BMI & Health Category Advisor agent.

Architecture (see README.md for the full diagram):

    User -> LLM (Ollama qwen2.5:3b) -> Agent -> calculate_bmi_category tool
         -> BMI result -> LLM -> Final response to user

Run modes:
    python app.py            -> interactive chat loop
    python app.py --demo     -> runs a scripted demo (valid + invalid input)
"""

import sys

from langchain.agents import create_agent
from langchain_ollama import ChatOllama

from bmi_tool import calculate_bmi_category

# ---------------------------------------------------------------------------
# 1. The LLM: a local Ollama model, no API key, no internet required.
#    temperature=0 keeps behaviour deterministic and predictable for a demo.
# ---------------------------------------------------------------------------
MODEL_NAME = "qwen2.5:3b"

llm = ChatOllama(
    model=MODEL_NAME,
    temperature=0,
)

# ---------------------------------------------------------------------------
# 2. The system prompt: tells the LLM *how* to behave, but NOT how to do
#    the math. The formula and category boundaries live only in bmi_tool.py.
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a BMI & Health Category Advisor.

Rules you must always follow:
1. Whenever a user gives (or you can infer) a weight in kilograms and a
   height in centimeters, call the `calculate_bmi_category` tool to get
   the BMI, category, and guidance. Never calculate BMI yourself -- the
   tool is the only source of truth for the number and category.
2. Pass the user's weight_kg and height_cm values to the tool exactly as
   given (convert units to kg/cm first if the user used other units).
3. If the tool returns a string starting with "Error:", explain the
   problem to the user in plain language and ask them to correct the
   input. Do not invent a BMI value yourself in that case.
4. Keep your final answer concise (2-4 sentences).
5. Never present BMI as a medical diagnosis. Always make clear that BMI
   is a general screening measure with known limitations (for example,
   it does not distinguish muscle mass from body fat).
6. Do not recommend medications or specific medical treatments. Keep any
   guidance general (e.g. "consider talking to a healthcare professional").
7. If the user has not given both weight and height, ask for the missing
   value instead of guessing.
"""

# ---------------------------------------------------------------------------
# 3. The agent: wires the LLM + the single tool + the system prompt together
#    using LangChain's create_agent(), which builds a LangGraph agent graph
#    under the hood (tool-calling loop is handled automatically).
# ---------------------------------------------------------------------------
agent = create_agent(
    model=llm,
    tools=[calculate_bmi_category],
    system_prompt=SYSTEM_PROMPT,
)


def ask_agent(user_message: str) -> str:
    """Send one message to the agent and return its final text reply."""
    result = agent.invoke({"messages": [{"role": "user", "content": user_message}]})
    final_message = result["messages"][-1]
    return final_message.content


def run_demo() -> None:
    """Scripted demonstration for a jury/viva: one valid, one invalid case."""
    print("=" * 70)
    print("DEMO 1: Valid input")
    print("=" * 70)
    q1 = "I weigh 70 kg and I am 175 cm tall. What is my BMI?"
    print(f"User: {q1}")
    print(f"Agent: {ask_agent(q1)}\n")

    print("=" * 70)
    print("DEMO 2: Invalid input (height = 0)")
    print("=" * 70)
    q2 = "I weigh 70 kg and my height is 0 cm."
    print(f"User: {q2}")
    print(f"Agent: {ask_agent(q2)}\n")


def run_interactive() -> None:
    """Simple REPL-style chat loop for manual testing / demonstration."""
    print("BMI & Health Category Advisor (type 'quit' to exit)")
    print("Example: 'I weigh 70 kg and I am 175 cm tall. What is my BMI?'\n")
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit"}:
            print("Goodbye!")
            break

        try:
            reply = ask_agent(user_input)
        except Exception as exc:  # pragma: no cover - safety net for the demo
            reply = (
                "Sorry, something went wrong talking to the local model. "
                f"Is Ollama running? (details: {exc})"
            )
        print(f"Agent: {reply}\n")


if __name__ == "__main__":
    if "--demo" in sys.argv:
        run_demo()
    else:
        run_interactive()
