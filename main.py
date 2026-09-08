"""AI E-Commerce Customer Support Agent with tools, memory, and return tracking."""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import ollama
from langchain_core.tools import tool


MODEL_NAME = "llama3.2"
DATA_DIR = Path(__file__).parent / "data"
ORDER_ID_PATTERN = re.compile(r"\bORD\d+\b", re.IGNORECASE)


def load_json(filename: str, default: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Load a JSON list from the data directory."""
    file_path = DATA_DIR / filename
    if not file_path.exists():
        if default is not None:
            return default
        raise FileNotFoundError(f"Missing required data file: {file_path}")
    with file_path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError(f"{filename} must contain a JSON list.")
    return data


products = load_json("products.json")
orders = load_json("orders.json")
return_requests = load_json("returns.json", default=[])


def save_return_requests() -> None:
    """Save return requests so duplicate return requests are avoided on the next run."""
    file_path = DATA_DIR / "returns.json"
    with file_path.open("w", encoding="utf-8") as file:
        json.dump(return_requests, file, indent=2)


def search_products(query: str) -> list[dict[str, Any]]:
    """Search products by name, category, or description."""
    query_words = {word for word in re.findall(r"[a-zA-Z0-9]+", query.lower()) if len(word) > 1}
    results: list[tuple[int, dict[str, Any]]] = []

    for product in products:
        searchable_text = " ".join(
            str(product.get(field, "")) for field in ("name", "category", "description")
        ).lower()
        score = sum(word in searchable_text for word in query_words)
        if score:
            results.append((score, product))

    results.sort(key=lambda item: item[0], reverse=True)
    return [product for _, product in results]


def format_products(results: list[dict[str, Any]]) -> str:
    if not results:
        return "No matching products found."

    return "\n".join(
        f"- {product['name']} (Rs.{product['price']}) - {product['description']} "
        f"[Stock: {product['stock']}]"
        for product in results[:5]
    )


def check_order_status(order_id: str) -> dict[str, Any] | None:
    order_id = order_id.upper()
    return next((order for order in orders if order["order_id"].upper() == order_id), None)


def format_order(order: dict[str, Any] | None) -> str:
    if not order:
        return "Order not found. Please check the order ID and try again."
    return (
        f"Order {order['order_id']} for {order['product_name']} is currently "
        f"'{order['status']}'. Expected delivery: {order['expected_delivery']}."
    )


def return_for_order(order_id: str) -> dict[str, Any] | None:
    return next(
        (request for request in return_requests if request["order_id"].upper() == order_id.upper()),
        None,
    )


@tool
def search_products_tool(query: str) -> str:
    """Search for products by name, category, or features. Use for product questions."""
    return format_products(search_products(query))


@tool
def check_order_status_tool(order_id: str) -> str:
    """Check an order's status or expected delivery date using an order ID such as ORD1002."""
    return format_order(check_order_status(order_id))


@tool
def initiate_return_tool(order_id: str) -> str:
    """Initiate a return for a delivered order. Use when a customer wants to return a product."""
    order = check_order_status(order_id)
    if not order:
        return "Order not found. Please provide a valid order ID."
    if order["status"].lower() != "delivered":
        return (
            f"Order {order['order_id']} cannot be returned yet because it is "
            f"currently '{order['status']}'. Returns are allowed after delivery."
        )

    existing_request = return_for_order(order["order_id"])
    if existing_request:
        return (
            f"A return request ({existing_request['return_id']}) has already been initiated "
            f"for order {order['order_id']} on {existing_request['requested_at']}."
        )

    request = {
        "return_id": f"RET{1001 + len(return_requests)}",
        "order_id": order["order_id"],
        "product_name": order["product_name"],
        "status": "Return requested",
        "requested_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    return_requests.append(request)
    save_return_requests()
    return (
        f"Return request {request['return_id']} initiated for order {order['order_id']} "
        f"({order['product_name']}). A confirmation email with pickup details will be sent within 24 hours."
    )


@tool
def recommend_products_tool(category: str) -> str:
    """Recommend in-stock products from a requested category, such as Electronics or Clothing."""
    results = [
        product
        for product in products
        if category.lower() in product.get("category", "").lower() and product.get("stock", 0) > 0
    ]
    if not results:
        return f"No in-stock recommendations are available in the '{category}' category."
    return format_products(results)


TOOLS = [
    search_products_tool,
    check_order_status_tool,
    initiate_return_tool,
    recommend_products_tool,
]

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "search_products_tool",
            "description": "Search products by name, category, or feature.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "What the customer wants to find"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_order_status_tool",
            "description": "Check the status or delivery of a customer order.",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string", "description": "Order ID, for example ORD1002"}},
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "initiate_return_tool",
            "description": "Request a return for a delivered order.",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string", "description": "Delivered order ID to return"}},
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recommend_products_tool",
            "description": "Recommend available products within a category.",
            "parameters": {
                "type": "object",
                "properties": {"category": {"type": "string", "description": "Product category, for example Electronics"}},
                "required": ["category"],
            },
        },
    },
]


conversation_history: list[dict[str, str]] = []
agent_state = {"last_order_id": None}


def remember_order_id(text: str) -> None:
    matches = ORDER_ID_PATTERN.findall(text)
    if matches:
        agent_state["last_order_id"] = matches[-1].upper()


def add_to_history(role: str, content: str) -> None:
    conversation_history.append({"role": role, "content": content})
    del conversation_history[:-10]  # Preserve only the recent conversation window.


def execute_tool(function_name: str, arguments: dict[str, Any]) -> str:
    """Execute only the approved local tools requested by the model."""
    if function_name in {"check_order_status_tool", "initiate_return_tool"}:
        arguments.setdefault("order_id", agent_state["last_order_id"])
        if not arguments.get("order_id"):
            return "Please provide the order ID, for example ORD1002."
        remember_order_id(str(arguments["order_id"]))

    handlers = {
        "search_products_tool": search_products_tool,
        "check_order_status_tool": check_order_status_tool,
        "initiate_return_tool": initiate_return_tool,
        "recommend_products_tool": recommend_products_tool,
    }
    handler = handlers.get(function_name)
    if not handler:
        return "Unknown tool requested."

    try:
        return str(handler.invoke(arguments))
    except Exception as error:
        return f"The requested operation could not be completed: {error}"


def run_agent(user_message: str) -> str:
    """Use Ollama to select a tool, execute it, and give a safe final response."""
    user_message = user_message.strip()
    if not user_message:
        return "Please enter a question."

    remember_order_id(user_message)
    recent_order_context = ""
    if agent_state["last_order_id"]:
        recent_order_context = (
            f" The most recently discussed order ID is {agent_state['last_order_id']}. "
            "Use it when the customer says 'it', 'that order', or 'my order'."
        )

    routing_messages = [
        {
            "role": "system",
            "content": (
                "You are an e-commerce customer-support agent. Use tools for factual questions "
                "about products, orders, returns, or recommendations." + recent_order_context
            ),
        },
        *conversation_history,
        {"role": "user", "content": user_message},
    ]

    try:
        response = ollama.chat(model=MODEL_NAME, messages=routing_messages, tools=TOOL_DEFINITIONS)
    except Exception as error:
        return (
            f"Could not connect to the Ollama model '{MODEL_NAME}'. "
            f"Run 'ollama list' and confirm the model is available. Details: {error}"
        )

    message = response["message"]
    tool_calls = message.get("tool_calls") or []
    if not tool_calls:
        answer = message.get("content") or "Please rephrase your question."
        add_to_history("user", user_message)
        add_to_history("assistant", answer)
        return answer

    tool_results = []
    for tool_call in tool_calls:
        function = tool_call["function"]
        function_name = function["name"]
        arguments = function.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {}

        result = execute_tool(function_name, arguments)
        print(f"[Agent used tool: {function_name} with args: {arguments}]")
        tool_results.append(f"{function_name}:\n{result}")

    verified_information = "\n\n".join(tool_results)
    final_prompt = f"""You are a helpful e-commerce customer-support assistant.
Customer question: {user_message}

Verified information from the local store system:
{verified_information}

Reply politely and clearly using only this verified information. Never invent a price, stock level, order status, delivery date, or return decision."""

    try:
        final_response = ollama.chat(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": final_prompt}],
        )
        answer = final_response["message"]["content"]
    except Exception:
        answer = verified_information

    add_to_history("user", user_message)
    add_to_history("assistant", answer)
    return answer


def print_tools() -> None:
    print("\n--- Tools Registered ---")
    for tool_item in TOOLS:
        print(f"- {tool_item.name}: {tool_item.description}")


def run_demo() -> None:
    print("\n=== TESTING AGENT ===")
    questions = [
        "Do you have any earbuds?",
        "What's the status of order ORD1002?",
        "Can I return it?",
        "Can you recommend some electronics?",
    ]
    for question in questions:
        print(f"\nUser: {question}")
        print("Agent:", run_agent(question))


def start_chat() -> None:
    print("\n=== LIVE CUSTOMER CHAT ===")
    print("Type 'exit' to close, or 'reset' to clear chat memory.")
    while True:
        user_message = input("\nYou: ").strip()
        if user_message.lower() == "exit":
            print("Agent: Thank you for contacting support. Goodbye!")
            break
        if user_message.lower() == "reset":
            conversation_history.clear()
            agent_state["last_order_id"] = None
            print("Agent: Conversation memory cleared.")
            continue
        print("Agent:", run_agent(user_message))


if __name__ == "__main__":
    print_tools()
    run_demo()
    start_chat()
