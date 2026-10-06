from litellm.llms.strands_decider.decisions.guardrail_translation.handler import DecisionsHandler
from litellm.types.utils import CallTypes

guardrail_translation_mappings = {
    CallTypes.decisions: DecisionsHandler,
    CallTypes.adecisions: DecisionsHandler,
}
