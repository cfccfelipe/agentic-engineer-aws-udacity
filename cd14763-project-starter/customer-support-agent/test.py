import asyncio

from main import invoke


async def run_local_test():
    print("🚀 Probando invoke() localmente con entorno: local...")

    # Payload simulado con la estructura que recibe el entrypoint de AgentCore
    payload = {
        "prompt": "Can you track order ORD-001?",
        "customer_id": "CUST-123",
        "session_id": "test-session-local-1",
    }

    try:
        response = await invoke(payload)
        print("\n================== RESULTADO LOCAL ==================")
        print(response)
        print("=====================================================")
    except Exception as e:
        print(f"\n❌ Error durante la ejecución local: {e}")


if __name__ == "__main__":
    asyncio.run(run_local_test())
