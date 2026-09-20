"""
Customer Support AI Agent — Official AgentCore Pattern with Managed MCP Context, Clean Routing & Strict Cost Protection
"""

# ── Imports ───────────────────────────────────────────────────────────────────
import argparse  # noqa: F401
import json
import logging
import os
import uuid
from typing import Any

import boto3
from bedrock_agentcore.memory import MemoryClient
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from mcp.client.streamable_http import streamable_http_client
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

# ── MCP Validation Patch for Pydantic v2 (Defensive CallToolResult Fallback) ──
try:
    from mcp.types import CallToolResult
    from mcp_types import methods as mcp_methods

    original_validate = getattr(mcp_methods, "validate_server_result", None)
    if original_validate:

        def patched_validate_server_result(method, version, data, surface=None):
            try:
                return original_validate(method, version, data, surface=surface)
            except Exception as e:
                if "ValidationError" in type(e).__name__ or "resultType" in str(e):
                    return CallToolResult(
                        content=[
                            {
                                "type": "text",
                                "text": "Backend returned an internal error response handled safely.",
                            }
                        ],
                        isError=True,
                    )
                raise e

        mcp_methods.validate_server_result = patched_validate_server_result
except Exception:
    pass

# ── Logging Setup ─────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.WARNING)
agent_logger = logging.getLogger("customersupportagent")

# ── Global Configuration ──────────────────────────────────────────────────────
os.environ["BYPASS_TOOL_CONSENT"] = "true"
os.environ["AGENTCORE_PROFILE_ID"] = "browser_profile_1x6ad-1lTDsfeNFy"
os.environ["BEDROCK_AGENTCORE_BROWSER_PROFILE_ID"] = "browser_profile_1x6ad-1lTDsfeNFy"
BROWSER_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
os.environ["AGENT_BROWSER_USER_AGENT"] = BROWSER_USER_AGENT

GATEWAY_URL = "https://customersupportgateway-vkbvhrtqpb.gateway.bedrock-agentcore.us-east-1.amazonaws.com/mcp"
KNOWLEDGE_BASE_ID = "0YZFL6UUQI"
REGION = "us-east-1"
MEMORY_ID = "CustomerSupportMemory-SP2UoJBNeT"

# CAMBIO DE MODELO: Usando Nova Micro para máxima velocidad, economía y menor latencia de timeout
AGENT_MODEL_ID = "amazon.nova-micro-v1:0"

# Prompt optimizado para evitar bucles
AGENT_SYSTEM_PROMPT = """
You are a professional customer support assistant. You have access to local tools, long-term customer memory, and remote tools via the MCP Gateway.
Maintain a helpful, concise, and purely factual tone. Never provide personal opinions.

TOOL USAGE DIRECTIVES:
1. Knowledge Base (`search_knowledge_base`):
   - Use only for product policies or return procedures. Never query repeatedly.

2. MCP Gateway Tools (`order-tracker` & `refund-processor`):
   - When checking an order (e.g., ORD-001), invoke `get_order` providing the exact parameter `order_id`.
   - If it fails, fall back to `get_customer_orders`. Do not retry failing calls more than once.

3. Loyalty Discount Calculator (`calculate_loyalty_discount`):
   - Use at most once per request.
"""

# ── Service Clients & Tools Instantiation ────────────────────────────────────
bedrock_model_instance = BedrockModel(model_id=AGENT_MODEL_ID)
memory_service_client = MemoryClient(region_name=REGION)
bedrock_runtime_client = boto3.client("bedrock-agent-runtime", region_name=REGION)

code_interpreter_utility = AgentCoreCodeInterpreter(region=REGION)
browser_utility = AgentCoreBrowser(region=REGION, session_timeout=60)

# ── Application Initialization ───────────────────────────────────────────────
agent_core_app = BedrockAgentCoreApp()


# ── Helper Functions ───────────────────────────────────────────────────────────
def fetch_memory_strategy_namespaces(
    memory_client: MemoryClient, memory_identifier: str
) -> dict:
    """Map strategy types to their corresponding namespace templates (cached)."""
    memory_strategy_cache = None
    if memory_strategy_cache is None:
        retrieved_strategies: list[dict[str, Any]] = (
            memory_client.get_memory_strategies(memory_id=memory_identifier)
        )
        memory_strategy_cache = {
            strategy_item["type"]: strategy_item["namespaces"][0]
            for strategy_item in retrieved_strategies
        }
    return memory_strategy_cache


# ── Memory Hook Class ─────────────────────────────────────────────────────────
class LongTermMemoryHook(HookProvider):
    """Event hook provider managing customer context retrieval and interaction storage."""

    def __init__(
        self,
        customer_actor_id: str,
        active_session_id: str,
        memory_client_instance: MemoryClient,
        target_memory_id: str,
    ):
        self.customer_actor_id = customer_actor_id
        self.active_session_id = active_session_id
        self.memory_client_instance = memory_client_instance
        self.target_memory_id = target_memory_id
        self.strategy_namespaces = fetch_memory_strategy_namespaces(
            memory_client_instance, target_memory_id
        )

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
                formatted_namespace = namespace_template.format(
                    actorId=self.customer_actor_id
                )
                retrieved_records = self.memory_client_instance.retrieve_memories(
                    memory_id=self.target_memory_id,
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
                self.memory_client_instance.create_event(
                    memory_id=self.target_memory_id,
                    actor_id=self.customer_actor_id,
                    session_id=self.active_session_id,
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


# ── Standalone Tools ──────────────────────────────────────────────────────────
@tool
def search_knowledge_base(search_query: str) -> str:
    """Search product specifications, return policies, warranty info, and loyalty details."""
    if not KNOWLEDGE_BASE_ID:
        return "Knowledge base not configured."

    try:
        retrieval_response = bedrock_runtime_client.retrieve(
            knowledgeBaseId=KNOWLEDGE_BASE_ID,
            retrievalQuery={"text": search_query},
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


# Variable global de control de seguridad (Circuit Breaker para costos)
_LOYALTY_EXECUTION_COUNT = 0


@tool
def calculate_loyalty_discount(
    loyalty_points: int,
    tier_name: str,
    order_subtotal: float,
    product_category: str = "standard",
) -> str:
    """Calculate loyalty discounts and points accrued."""
    global _LOYALTY_EXECUTION_COUNT
    _LOYALTY_EXECUTION_COUNT += 1

    # PROTECCIÓN ABSOLUTA: Si se intenta llamar más de 2 veces, se bloquea por completo
    if _LOYALTY_EXECUTION_COUNT > 2:
        return json.dumps({
            "error": "Execution limit reached for loyalty calculation.",
            "final_total": order_subtotal,
            "total_savings": 0.0,
        })

    import math

    earn_rates = {"standard": 1, "device": 2, "fresh": 5}
    tier_rates = {"Silver": 0.00, "Gold": 0.10, "Platinum": 0.15}

    max_points_discount = order_subtotal * 0.50
    max_usable_points = int(max_points_discount * 100)
    points_to_use = min(int(loyalty_points), max_usable_points)
    points_redeemed = math.floor(points_to_use / 500) * 500
    points_discount = points_redeemed / 100.0

    subtotal_after_points = order_subtotal - points_discount
    tier_discount_rate = tier_rates.get(tier_name, 0.0)
    tier_discount = round(subtotal_after_points * tier_discount_rate, 2)

    final_total = round(subtotal_after_points - tier_discount, 2)
    total_savings = round(points_discount + tier_discount, 2)
    earn_rate = earn_rates.get(product_category, 1)
    points_earned = int(final_total * earn_rate)
    remaining_points = int(loyalty_points) - points_redeemed + points_earned

    return json.dumps({
        "original_total": order_subtotal,
        "points_redeemed": points_redeemed,
        "points_discount": points_discount,
        "tier": tier_name,
        "tier_discount": tier_discount,
        "final_total": final_total,
        "total_savings": total_savings,
        "points_earned": points_earned,
        "remaining_points": remaining_points,
    })


# ── Agent Core Execution Entrypoint ───────────────────────────────────────────
@agent_core_app.entrypoint
async def invoke(payload: dict, context: Any = None) -> str:
    """Primary endpoint invoked by AgentCore runtime for each incoming payload."""
    try:
        incoming_user_prompt = payload.get("prompt", "")
        customer_actor_id = payload.get("customer_id") or payload.get(
            "actor_id", "default_customer"
        )
        active_session_id = payload.get("session_id") or str(uuid.uuid4())

        active_memory_hook = LongTermMemoryHook(
            customer_actor_id=customer_actor_id,
            active_session_id=active_session_id,
            memory_client_instance=memory_service_client,
            target_memory_id=MEMORY_ID,
        )

        built_in_tools = [
            search_knowledge_base,
            calculate_loyalty_discount,
            browser_utility.browser,
        ]

        agent_invocation_response = None

        if GATEWAY_URL:
            try:
                mcp_client = MCPClient(lambda: streamable_http_client(GATEWAY_URL))

                with mcp_client:
                    mcp_gateway_tools = mcp_client.list_tools_sync()
                    consolidated_tools = built_in_tools + list(mcp_gateway_tools or [])

                    # LÍMITE ESTRICTO DE 2 ITERACIONES PARA EVITAR BUCLES Y COSTOS
                    support_agent_instance = Agent(
                        model=bedrock_model_instance,
                        tools=consolidated_tools,  # type: ignore[arg-type]
                        hooks=[active_memory_hook],
                        system_prompt=AGENT_SYSTEM_PROMPT,
                        max_iterations=2,
                    )

                    agent_invocation_response = support_agent_instance(
                        incoming_user_prompt
                    )

            except Exception as mcp_err:
                agent_logger.warning("MCP Gateway context error: %s", mcp_err)

        if agent_invocation_response is None:
            support_agent_instance = Agent(
                model=bedrock_model_instance,
                tools=built_in_tools,  # type: ignore[arg-type]
                hooks=[active_memory_hook],
                system_prompt=AGENT_SYSTEM_PROMPT,
                max_iterations=2,
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


def main():
    agent_core_app.run()


if __name__ == "__main__":
    main()
