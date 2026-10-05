from .fsm import (JSONStateMachine, JSONState)
from ..llm_sdk.llm_sdk import Small_LLM_Model


def constrained_decoding(model: Small_LLM_Model, fsm: JSONStateMachine,
                         initial_input_ids: list[int]) -> str:

    input_ids = initial_input_ids
    while fsm.state != JSONState.END:
        allowed_tokens = fsm.get_allowed_token_ids()
        if not allowed_tokens:
            raise RuntimeError("FSM reached a dead end with no allowed tokens")
        logits = model.get_logits_from_input_ids(input_ids)
        next_token_id = max(
            allowed_tokens, key=lambda token_id: logits[token_id]
        )
        token_str = model.decode(next_token_id)
        fsm.update_state(token_str)
        input_ids.append(next_token_id)

    return fsm.get_final_output(model, input_ids, initial_input_ids)
