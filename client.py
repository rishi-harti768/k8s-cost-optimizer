"""K8s Cost Optimizer Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import Action as K8sCostOptimizerAction, Observation as K8sCostOptimizerObservation


class KubeCostEnvClient(
    EnvClient[K8sCostOptimizerAction, K8sCostOptimizerObservation, State]
):
    """
    Client for the KubeCost Environment.
    """

    def _step_payload(self, action: K8sCostOptimizerAction) -> Dict:
        return {"action": action.model_dump()}

    def _parse_result(self, payload: Dict) -> StepResult[K8sCostOptimizerObservation]:
        obs_data = payload.get("observation", {})
        observation = K8sCostOptimizerObservation(**obs_data)

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id", "unknown"),
            step_count=payload.get("step_count", 0),
        )
