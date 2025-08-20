from typing import Any
from pydantic import ValidationError

from trinity.utils.log import get_logger
from trinity.common.workflows.envs.email_searcher.utils import get_rubric_state

try:
    import agentscope
except ImportError as e:
    error_message = f"AgentScope is not installed. Please install the agentscope framework first before running the workflow. Error: {str(e)}"
    print(error_message)
    agentscope = None

from agentscope.agents import ReActAgentV2 
from agentscope.service import ServiceResponse, ServiceExecStatus

logger = get_logger(__name__)



class ReActAgent(ReActAgentV2):
    """
    A customized ReAct agent that overrides the generate_response method by return_final_answer
    Ref: https://github.com/OpenPipe/ART/blob/main/dev/art-e/art_e/rollout.py#L260
    """
    def generate_response(
        self,
        response: str,  # pylint: disable=unused-argument
        **kwargs: Any,
    ) -> ServiceResponse:
        """Return the final answer to the user's query.
        It should be called with the answer and the sources. If you cannot find the answer, you should return "I don't know" with an empty list of sources.

        Args:
            response (`str`):
                The response to the user.
        """
        logger.info(f"{kwargs = } ")
        if self._current_structured_model:
            try:
                # need: attempted_answer, returned_i_dont_know, sources
                if "I don't know" in response:
                    returned_i_dont_know = True  # TODO: may not need this? Or does this really work?
                rubric = {
                    "attempted_answer": True,
                    "returned_i_dont_know": returned_i_dont_know,
                    **get_rubric_state(),
                }
                self._current_structured_model.model_validate(rubric)  # in-place operation
                self._current_structured_output = rubric
            except ValidationError as e:
                return ServiceResponse(
                    status=ServiceExecStatus.ERROR,
                    content=f"Validation error: {e.errors()}",
                )

        return ServiceResponse(
            status=ServiceExecStatus.SUCCESS,
            content="Success",
        )
