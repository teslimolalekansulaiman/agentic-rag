from nemoguardrails import LLMRails, RailsConfig
from crewai import Agent, Task, Crew, Process

from src.common.config import settings

# --- Initialize NeMo Guardrails with Ollama backend ---
config = RailsConfig.from_path(f"{settings.RAIL_CONFIGURATION_DIR}/config.yml")  # 👈 path to directory that has config.yml
guardrails = LLMRails(config)


# === Guarded Agent ===
from nemoguardrails import LLMRails, RailsConfig


class GuardedAgent(Agent):
    """A CrewAI Agent that routes I/O through NeMo Guardrails."""

    def _guard_input(self, text: str) -> tuple[bool, str]:
        """Check if input is safe. Returns (is_safe, output_or_error)."""
        try:
            result = guardrails.generate(
                messages=[{"role": "user", "content": text}]
            )

            # Check if blocked
            if isinstance(result, dict) and result.get("is_blocked", False):
                return False, f"❌ Input blocked: {result.get('block_reason', 'Safety violation')}"

            return True, text
        except Exception as e:
            return False, f"Validation error: {str(e)}"

    def _guard_output(self, text: str) -> tuple[bool, str]:
        """Check if output is safe. Returns (is_safe, output)."""
        try:
            result = guardrails.generate(
                messages=[{"role": "assistant", "content": text}]
            )

            if isinstance(result, dict) and result.get("is_blocked", False):
                return False, f"❌ Output blocked: {result.get('block_reason', 'Safety violation')}"

            return True, text
        except Exception as e:
            return False, f"Validation error: {str(e)}"

    def execute_task(self, task, *args, **kwargs) -> str:
        """Override execute_task (the actual execution method)."""
        # Check input
        is_safe, message = self._guard_input(str(task.description))
        if not is_safe:
            return message

        # Execute
        try:
            result = super().execute_task(task, *args, **kwargs)
        except Exception as e:
            return f"❌ Execution error: {str(e)}"

        # Check output
        is_safe, output = self._guard_output(str(result))
        if not is_safe:
            return output

        return result


