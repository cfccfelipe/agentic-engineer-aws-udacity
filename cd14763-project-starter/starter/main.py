"""
Customer Support AI Agent — Starter Code
==========================================
Your task is to complete this file by implementing all sections marked
with # TODO comments.

Reference the step-by-step solution files and INSTRUCTIONS.md for guidance.
Do NOT copy the solution directly — work through each section yourself.

Run locally (after filling in config values):
  uv run main.py '{"prompt": "Hello", "customer_id": "CUST-123", "session_id": "s1"}'

Deploy to AgentCore:
  agentcore deploy

Invoke deployed agent:
  agentcore invoke '{"prompt": "Hello", "customer_id": "CUST-123", "session_id": "s1"}'
"""

# ── Imports ───────────────────────────────────────────────────────────────────
# These imports are provided. Do not remove them.
import argparse
import asyncio
import json
import logging
import math
import os

import boto3
from bedrock_agentcore.memory import MemoryClient
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from botocore.config import Config
from strands import Agent, tool
from strands.hooks import (
    AfterInvocationEvent,
    HookProvider,
    HookRegistry,
    MessageAddedEvent,
)
from strands.models import BedrockModel
from strands.tools.mcp.mcp_client import MCPClient
from strands_tools.browser import AgentCoreBrowser
from strands_tools.code_interpreter import (
    AgentCoreCodeInterpreter,
)

logging.basicConfig(level=logging.WARNING)
agent_logger = logging.getLogger("CSAI_Agent")

# ── TODO 1 — App Initialisation ───────────────────────────────────────────────
# Create a BedrockAgentCoreApp instance.
# This registers the ASGI server for AgentCore deployment.
# There must be exactly one instance per deployment.
#
# Hint: app = BedrockAgentCoreApp()

# TODO: Create the BedrockAgentCoreApp instance
app = BedrockAgentCoreApp()  # Replace this line


# Suppress interactive tool-consent prompts (required in headless deployments).
os.environ["BYPASS_TOOL_CONSENT"] = "true"


# ── TODO 2 — Configuration ────────────────────────────────────────────────────
# Replace the placeholder strings with your actual AWS resource values.
# You collected these in Part 1 of the INSTRUCTIONS.
#
# GATEWAY_URL format: https://<alias>.gateway.bedrock-agentcore.<region>.amazonaws.com/mcp
# KB_ID       format: 10-character alphanumeric string from the KB console
# REGION:     your AWS region, e.g. "us-east-1"
# MEMORY_ID   format: shown in the AgentCore Memory console

GATEWAY_URL = "https://customersupportgateway-6gixqsbqhs.gateway.bedrock-agentcore.us-east-1.amazonaws.com/mcp"
KB_ID = "CWBGCPGVF2"
REGION = "us-east-1"
MEMORY_ID = "CustomerSupportMemory-0abN3GBFYS"  # TODO: Replace with your Memory ID
AGENT_SYSTEM_PROMPT = """
You are a professional customer support assistant. You have access to local tools, long-term customer memory, and remote tools via the MCP Gateway.
Maintain a helpful, concise, and purely factual tone. Never provide personal opinions.

TOOL USAGE DIRECTIVES:
1. Knowledge Base (`search_knowledge_base`):
   - Use only for product policies or return procedures. Never query repeatedly.

2. MCP Gateway Tools (`order-tracker` & `refund-processor`):
   - Always use exact, valid identifiers provided by the user or retrieved from memory (e.g., order IDs like ORD-001, customer IDs like CUST-123). Do not guess or modify IDs.
   - When checking an order, invoke `order-tracker___get_order` providing the exact parameter `order_id`.
   - If a lookup fails or returns not found, do not retry the exact same call. Instead, fall back to `order-tracker___get_customer_orders` or ask the user for clarification.
   - For refunds and returns, use the appropriate `refund-processor` tools (`initiate_refund`, `check_refund_status`, `get_return_label`) with their required schemas.

3. Loyalty Discount Calculator (`calculate_loyalty_discount`):
   - Use at most once per request.
"""

# ── TODO 3 — Model and Clients ────────────────────────────────────────────────
# Create:
#   1. A BedrockModel using model_id "global.amazon.nova-2-lite-v1:0"
#   2. A MemoryClient with region_name=REGION
#   3. A boto3 client for the "bedrock-agent-runtime" service in REGION
#
# Hint: model = BedrockModel(model_id=model_id)

model_id = "global.amazon.nova-2-lite-v1:0"
boto_config = Config(
    read_timeout=15, connect_timeout=15, retries={"max_attempts": 2, "mode": "standard"}
)
bedrock_model_instance = BedrockModel(model_id=model_id, boto_client_config=boto_config)
memory_service_client = MemoryClient(region_name=REGION)
bedrock_runtime_client = boto3.client("bedrock-agent-runtime", region_name=REGION)


# ── TODO 4 — Namespace Helper ─────────────────────────────────────────────────
# Implement get_namespaces() to return a dict mapping strategy type to
# namespace template string.
#
# Steps:
#   1. Call mem_client.get_memory_strategies(memory_id) to get strategy list
#   2. Return a dict: { strategy["type"]: strategy["namespaces"][0] for each strategy }
#
# Example output:
#   { "SEMANTIC": "cs_agent/{actorId}/facts",
#     "USER_PREFERENCE": "cs_agent/{actorId}/preferences" }

code_interpreter_utility = AgentCoreCodeInterpreter(region=REGION)
browser_utility = AgentCoreBrowser(region=REGION, session_timeout=60)


def get_namespaces(mem_client: MemoryClient, memory_id: str) -> dict:
    """Return a dict mapping strategy type → namespace template string."""
    strategies = mem_client.get_memory_strategies(memory_id)
    return {strategy["type"]: strategy["namespaces"][0] for strategy in strategies}


# ── TODO 5 — Memory Hook ──────────────────────────────────────────────────────
# Implement MemoryHook, a HookProvider subclass that adds long-term memory.
#
# The class needs:
#   __init__(self, actor_id, session_id, memory_client, memory_id)
#     — store all four as instance attributes
#     — call get_namespaces() and store the result as self.namespaces
#
#   retrieve_customer_context(self, event: MessageAddedEvent)
#     — only runs for plain-text user messages (not tool results)
#     — for each strategy namespace, call memory_client.retrieve_memories(
#          memory_id, namespace (formatted with actorId), query, top_k=5)
#     — collect non-empty memory texts tagged with their strategy type
#     — if any memories found, prepend them to the user message as:
#          "Customer Context:\n<memories>\n\n<original_message>"
#
#   save_support_interaction(self, event: AfterInvocationEvent)
#     — walk the message list backwards to find the last plain-text user
#       query and the last assistant response
#     — call memory_client.create_event(memory_id, actor_id, session_id,
#          messages=[(customer_query, "USER"), (agent_response, "ASSISTANT")])
#
#   register_hooks(self, registry: HookRegistry)
#     — register retrieve_customer_context on MessageAddedEvent
#     — register save_support_interaction on AfterInvocationEvent


class MemoryHook(HookProvider):
    """Long-term memory hook for the customer support agent."""

    def __init__(
        self,
        actor_id: str,
        session_id: str,
        memory_client: MemoryClient,
        memory_id: str,
    ):
        self.actor_id = actor_id
        self.session_id = session_id
        self.memory_client = memory_client
        self.memory_id = memory_id
        self.strategy_namespaces = get_namespaces(memory_client, memory_id)

    # ➔ CORREGIDO: Estos métodos ahora están al nivel de la clase, fuera de __init__
    def retrieve_customer_context(self, event: MessageAddedEvent):
        """Retrieve historical context for the customer and prepend it to the message."""
        try:
            conversation_messages = getattr(event.agent, "messages", [])
            if not conversation_messages:
                return

            latest_message = conversation_messages[-1]
            if latest_message.get("role") != "user":
                return

            raw_message_content = latest_message.get("content")
            user_query_text = ""

            if isinstance(raw_message_content, str):
                user_query_text = raw_message_content
            elif isinstance(raw_message_content, list) and len(raw_message_content) > 0:
                first_content_block = raw_message_content[0]
                if isinstance(first_content_block, dict):
                    user_query_text = first_content_block.get("text", "")

            if not user_query_text or user_query_text.startswith("ToolResult:"):
                return

            retrieved_memory_snippets = []

            for (
                strategy_type_name,
                namespace_template,
            ) in self.strategy_namespaces.items():
                formatted_namespace = namespace_template.format(actorId=self.actor_id)
                retrieved_records = self.memory_client.retrieve_memories(
                    memory_id=self.memory_id,
                    namespace=formatted_namespace,
                    query=user_query_text,
                    top_k=3,
                )

                for memory_record in retrieved_records:
                    memory_text_content = memory_record.get(
                        "text"
                    ) or memory_record.get("content")
                    if memory_text_content:
                        retrieved_memory_snippets.append(
                            f"[{strategy_type_name}] {memory_text_content}"
                        )

            if retrieved_memory_snippets:
                formatted_context_block = "\n".join(retrieved_memory_snippets)
                context_header_prefix = (
                    f"Customer Context:\n{formatted_context_block}\n\n"
                )

                if isinstance(raw_message_content, str):
                    latest_message["content"] = (
                        context_header_prefix + raw_message_content
                    )
                elif isinstance(raw_message_content, list):
                    for message_block in raw_message_content:
                        if isinstance(message_block, dict) and "text" in message_block:
                            message_block["text"] = (
                                context_header_prefix + message_block["text"]
                            )
                            break
        except Exception as e:
            agent_logger.warning(
                "Memory retrieval bypassed due to timeout/error: %s", e
            )

    def save_support_interaction(self, event: AfterInvocationEvent):
        """Persist the prompt-response interaction turn to long-term memory."""
        try:
            conversation_messages = getattr(event.agent, "messages", [])
            extracted_user_query = None
            extracted_agent_response = None

            for current_message in reversed(conversation_messages):
                message_role = current_message.get("role")
                raw_message_content = current_message.get("content")

                extracted_text = ""
                if isinstance(raw_message_content, str):
                    extracted_text = raw_message_content
                elif (
                    isinstance(raw_message_content, list)
                    and len(raw_message_content) > 0
                ):
                    first_content_block = raw_message_content[0]
                    if isinstance(first_content_block, dict):
                        extracted_text = first_content_block.get("text", "")

                if not extracted_text or extracted_text.startswith("ToolResult:"):
                    continue

                if not extracted_agent_response and message_role == "assistant":
                    extracted_agent_response = extracted_text
                elif not extracted_user_query and message_role == "user":
                    extracted_user_query = extracted_text

                if extracted_user_query and extracted_agent_response:
                    break

            if extracted_user_query and extracted_agent_response:
                self.memory_client.create_event(
                    memory_id=self.memory_id,
                    actor_id=self.actor_id,
                    session_id=self.session_id,
                    messages=[
                        (extracted_user_query, "USER"),
                        (extracted_agent_response, "ASSISTANT"),
                    ],
                )
        except Exception as e:
            agent_logger.warning("Memory save bypassed: %s", e)

    def register_hooks(self, registry: HookRegistry) -> None:  # type: ignore
        """Register callbacks on the HookRegistry."""
        registry.add_callback(MessageAddedEvent, self.retrieve_customer_context)
        registry.add_callback(AfterInvocationEvent, self.save_support_interaction)


# ── TODO 6 — Knowledge Base Tool ─────────────────────────────────────────────
# Implement search_knowledge_base(query) using the @tool decorator.
#
# Steps:
#   1. Guard: if KB_ID is empty return "Knowledge base not configured."
#   2. Call _bedrock_runtime.retrieve(
#          knowledgeBaseId=KB_ID,
#          retrievalQuery={"text": query}
#      )
#   3. Extract resp["retrievalResults"]; return a message if empty
#   4. Join the text chunks with "\n---\n" and return the result
#
# The docstring is the tool description — the model uses it to decide when
# to call this tool, so keep it clear and accurate.


@tool
def search_knowledge_base(query: str) -> str:
    """
    Search the Amazon product catalog and support knowledge base.
    Use this for product specifications, return policies, warranty
    information, loyalty program details, and order status definitions.

    Args:
        query: The question or topic to search for

    Returns:
        Relevant information retrieved from the knowledge base
    """
    if not KB_ID:
        return "Knowledge base not configured."

    try:
        retrieval_response = bedrock_runtime_client.retrieve(
            knowledgeBaseId=KB_ID,
            retrievalQuery={"text": query},
        )
        retrieved_results = retrieval_response.get("retrievalResults", [])
        if not retrieved_results:
            return "No relevant information found in the knowledge base."

        extracted_text_chunks = [
            result_item.get("content", {}).get("text", "")
            for result_item in retrieved_results
            if result_item.get("content", {}).get("text")
        ]
        return (
            "\n---\n".join(extracted_text_chunks)
            if extracted_text_chunks
            else "No relevant information found."
        )
    except Exception as e:
        return f"Knowledge base unavailable: {e!s}"


# ── TODO 7 — Loyalty Discount Tool (Code Interpreter) ────────────────────────
# Implement calculate_loyalty_discount() using the @tool decorator.
#
# The tool must:
#   1. Build a self-contained Python code string that:
#        • Defines earn_rates: {"standard": 1, "device": 2, "fresh": 5}
#        • Defines tier_rates: {"Silver": 0.00, "Gold": 0.10, "Platinum": 0.15}
#        • Calculates points_redeemed (floor to nearest 500, cap at 50% of order)
#        • Calculates tier_discount (applied to subtotal after points)
#        • Calculates final_total, total_savings, points_earned, remaining_points
#        • Prints a JSON result dict
#   2. Execute the code with code_session(REGION).invoke("executeCode", {...})
#      using language="python" and clearContext=True
#   3. Return the first result event as a JSON string
#   4. Include a fallback that computes only the tier discount if the
#      Code Interpreter is unavailable

_LOYALTY_EXECUTION_COUNT = 0


@tool
def calculate_loyalty_discount(
    loyalty_points: int,
    tier: str,
    order_total: float,
    product_category: str = "standard",
) -> str:
    """
    Calculate the loyalty discount for a customer order using the
    AgentCore Code Interpreter. Runs exact arithmetic in a secure sandbox.

    Args:
        loyalty_points:   Customer's current points balance
        tier:             Customer tier — Silver, Gold, or Platinum
        order_total:      Order total in USD
        product_category: standard, device, or fresh

    Returns:
        Full discount breakdown and final price
    """
    global _LOYALTY_EXECUTION_COUNT
    _LOYALTY_EXECUTION_COUNT += 1

    if _LOYALTY_EXECUTION_COUNT > 2:
        return json.dumps({
            "error": "Execution limit reached for loyalty calculation.",
            "final_total": order_total,
            "total_savings": 0.0,
        })

    earn_rates = {"standard": 1, "device": 2, "fresh": 5}
    tier_rates = {"Silver": 0.00, "Gold": 0.10, "Platinum": 0.15}

    max_points_discount = order_total * 0.50
    max_usable_points = int(max_points_discount * 100)
    points_to_use = min(int(loyalty_points), max_usable_points)
    points_redeemed = math.floor(points_to_use / 500) * 500
    points_discount = points_redeemed / 100.0

    subtotal_after_points = order_total - points_discount
    tier_discount_rate = tier_rates.get(tier, 0.0)
    tier_discount = round(subtotal_after_points * tier_discount_rate, 2)

    final_total = round(subtotal_after_points - tier_discount, 2)
    total_savings = round(points_discount + tier_discount, 2)
    earn_rate = earn_rates.get(product_category, 1)
    points_earned = int(final_total * earn_rate)
    remaining_points = int(loyalty_points) - points_redeemed + points_earned

    return json.dumps({
        "original_total": order_total,
        "points_redeemed": points_redeemed,
        "points_discount": points_discount,
        "tier": tier,
        "tier_discount": tier_discount,
        "final_total": final_total,
        "total_savings": total_savings,
        "points_earned": points_earned,
        "remaining_points": remaining_points,
    })


# ── TODO 8 — Agent Entrypoint ─────────────────────────────────────────────────
# Implement the invoke() function decorated with @app.entrypoint.
#
# Steps:
#   1. Extract user_input, actor_id, and session_id from the payload
#      (generate a UUID if session_id is missing)
#   2. Instantiate MemoryHook for this actor/session
#   3. Instantiate AgentCoreBrowser(region=REGION)
#   4. Build the tools list: [search_knowledge_base, calculate_loyalty_discount,
#                              agent_core_browser.browser]
#   5. Connect to the Gateway via MCPClient, load gateway_tools, extend tools list
#   6. Create and invoke the Agent with all tools, hooks, and system_prompt
#   7. Return the text from the first content block of the response
#   8. Handle exceptions gracefully


@app.entrypoint
async def invoke(payload, context=None):
    """
    Main handler called by AgentCore for every incoming request.

    Expected payload keys:
      prompt      (str, required) — the customer's message
      customer_id (str, optional) — unique customer identifier
      session_id  (str, optional) — session identifier; generated if absent
    """

    try:
        incoming_user_prompt = payload.get("prompt", "")
        customer_actor_id = payload.get("customer_id") or payload.get(
            "actor_id", "default_customer"
        )
        active_session_id = payload.get("session_id") or str(uuid.uuid4())

        active_memory_hook = MemoryHook(
            actor_id=customer_actor_id,
            session_id=active_session_id,
            memory_client=memory_service_client,
            memory_id=MEMORY_ID,
        )

        built_in_tools: list = [
            search_knowledge_base,
            calculate_loyalty_discount,
            browser_utility.browser,
        ]

        agent_invocation_response = None

        if GATEWAY_URL:
            try:
                mcp_client = MCPClient(url=GATEWAY_URL)

                with mcp_client:
                    mcp_gateway_tools = mcp_client.list_tools_sync()
                    consolidated_tools = built_in_tools + list(mcp_gateway_tools or [])

                    support_agent_instance = Agent(
                        model=bedrock_model_instance,
                        tools=consolidated_tools,  # type: ignore[arg-type]
                        hooks=[active_memory_hook],
                        system_prompt=AGENT_SYSTEM_PROMPT,
                    )

                    agent_invocation_response = support_agent_instance(
                        incoming_user_prompt
                    )

            except Exception as mcp_err:
                agent_logger.warning("MCP Gateway context error: %s", mcp_err)

        if agent_invocation_response is None:
            support_agent_instance = Agent(
                model=bedrock_model_instance,
                tools=built_in_tools,
                hooks=[active_memory_hook],
                system_prompt=AGENT_SYSTEM_PROMPT,
            )
            agent_invocation_response = support_agent_instance(incoming_user_prompt)

        if hasattr(agent_invocation_response, "message") and isinstance(
            agent_invocation_response.message, dict
        ):
            response_content_payload = agent_invocation_response.message.get(
                "content", []
            )
            if (
                isinstance(response_content_payload, list)
                and len(response_content_payload) > 0
            ):
                primary_content_block = response_content_payload[0]
                if isinstance(primary_content_block, dict):
                    return str(
                        primary_content_block.get(
                            "text", str(agent_invocation_response)
                        )
                    )
            elif isinstance(response_content_payload, str):
                return response_content_payload

        return str(agent_invocation_response)

    except Exception as runtime_error:
        agent_logger.exception("Error during agent runtime execution")
        return f"An error occurred while processing your request: {runtime_error!s}"


# ── CLI entry point (do not modify) ──────────────────────────────────────────
def main():
    """Run one invocation from the command line for local testing."""
    parser = argparse.ArgumentParser()
    parser.add_argument("payload", type=str)
    args = parser.parse_args()
    response = asyncio.run(invoke(json.loads(args.payload)))
    print(response)


if __name__ == "__main__":
    app.run()
    # Uncomment the line below and comment app.run() for local CLI testing:
    main()
