import os
import logging
from phoenix.otel import register

def setup_phoenix_tracing() -> None:
    phoenix_host = os.getenv("PHOENIX_HOST", "http://localhost:6006")
    phoenix_endpoint = f"{phoenix_host}/v1/traces"
    os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = phoenix_endpoint

    try:
        global tracer_provider
        tracer_provider = register(
            project_name="rag-api",
            endpoint=phoenix_endpoint,
            auto_instrument=True
        )
        logging.basicConfig(level=logging.INFO)
        logging.info(f"✅ Phoenix tracing sending to {phoenix_endpoint}")
    except Exception as e:
        logging.warning(f"⚠️ Could not initialize Phoenix tracing: {e}")
