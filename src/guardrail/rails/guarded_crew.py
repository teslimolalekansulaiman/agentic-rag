from crewai import Crew
from typing import Any, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class GuardedCrew:
    """Crew wrapper that validates inputs and outputs using NeMo Guardrails."""

    def __init__(self, crew: Crew, guardrails):
        if not guardrails:
            raise ValueError("GuardedCrew requires a guardrails instance.")
        self.crew = crew
        self.guardrails = guardrails

    def _run_guardrails(self, role: str, text: str) -> Dict[str, Any]:
        """Internal helper to call the guardrails engine."""
        try:
            response = self.guardrails.generate(messages=[{"role": role, "content": text}])

            # Dict-style response
            if isinstance(response, dict):
                if response.get("is_blocked") or response.get("blocked_by_guardrail"):
                    return {"is_safe": False, "reason": response.get("block_reason", "Blocked by guardrail")}
                if response.get("validation_passed") is False:
                    return {"is_safe": False, "reason": response.get("message", "Validation failed")}

            # String-style response (fallback)
            if isinstance(response, str) and any(
                word in response.lower() for word in ["blocked", "refused", "not allowed", "sorry"]
            ):
                return {"is_safe": False, "reason": response}

            return {"is_safe": True}
        except Exception as e:
            return {"is_safe": False, "reason": f"Validation error: {e}"}

    def _guard_input(self, text: str) -> Dict[str, Any]:
        """Validate user input."""
        logger.debug(f"Validating input: {text[:60]}...")
        result = self._run_guardrails("user", text)
        if not result["is_safe"]:
            logger.warning(f"Input blocked: {result['reason']}")
        else:
            logger.debug("Input safe.")
        return result

    def _guard_output(self, text: str) -> Dict[str, Any]:
        """Validate model output."""
        logger.debug("Validating output...")
        result = self._run_guardrails("assistant", text)
        if not result["is_safe"]:
            logger.warning(f"Output blocked: {result['reason']}")
        else:
            logger.debug("Output safe.")
        return result


    def _prepare_inputs(self, inputs: Optional[Dict[str, Any]]) -> Optional[str]:
        """Extract user input text for validation."""
        if not inputs:
            return None
        return inputs.get("query")

    def _execute_crew(self, inputs: Optional[Dict[str, Any]]) -> Any:
        """Call the underlying crew safely."""
        try:
            logger.info("🚀 Executing Crew task...")
            return self.crew.kickoff(inputs=inputs)
        except Exception as e:
            logger.error(f" Crew execution failed: {e}", exc_info=True)
            return {"error": f"Execution failed: {e}", "blocked": False}

    def _handle_blocked(self, reason: str, phase: str) -> Dict[str, Any]:
        """Uniform blocked response."""
        return {"error": f" {phase} blocked: {reason}", "blocked": True, "reason": reason}


    def kickoff(self, inputs: Optional[Dict[str, Any]] = None) -> Any:
        """Full guarded execution pipeline."""
        user_text = self._prepare_inputs(inputs)

        # 1️⃣ Guard input
        if user_text:
            check = self._guard_input(user_text)
            if not check["is_safe"]:
                return self._handle_blocked(check["reason"], "Input")

        # 2️⃣ Run Crew
        result = self._execute_crew(inputs)
        if isinstance(result, dict) and result.get("error"):
            return result  # already handled error

        # 3️⃣ Guard output
        if isinstance(result, str):
            check = self._guard_output(result)
            if not check["is_safe"]:
                return self._handle_blocked(check["reason"], "Output")

        logger.info("Crew output passed safety checks")
        return result

    def __getattr__(self, name):
        return getattr(self.crew, name)
