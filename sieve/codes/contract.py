"""The /v1/systemone request contract.

Request models accept exactly what the official spec accepts (plus a lenient `state: null`, which the JS SDK can send),
so pydantic's discriminated-union errors reproduce FastAPI's 422 bodies with the tag in `loc`
(e.g. ["body","questions","urgency","score","criteria"]). Limits the spec does not encode (255 options, 10 score levels,
32k / 64k tokens) are checked after parsing and raised as `ContractError`, rendered as `{"detail":{...}}`; the server
also applies the input rules every Sieve model shares (sieve.api.validate_questions) as 400 invalid_request.
"""
from __future__ import annotations

from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field

Content = Union[str, dict[str, Any], list[Any]]

MAX_CHOICE_OPTIONS = 255
MAX_SCORE_LEVELS = 10
MAX_BRANCH_TOKENS = 32_768      # prefix (state) + the single longest question
MAX_REQUEST_TOKENS = 65_536     # prefix (state, once) + every question


class NoulCriteria(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)
    true: Content | None = None
    false: Content | None = None


class NoulQuestion(BaseModel):
    model_config = ConfigDict(extra="ignore")
    type: Literal["noul"]
    instructions: Content | None = None
    criteria: NoulCriteria | None = None


class ChoiceQuestion(BaseModel):
    model_config = ConfigDict(extra="ignore")
    type: Literal["choice"]
    instructions: Content | None = None
    criteria: dict[str, Content | None]


class ScoreQuestion(BaseModel):
    model_config = ConfigDict(extra="ignore")
    type: Literal["score"]
    instructions: Content | None = None
    criteria: Annotated[list[Content], Field(min_length=1)]


Question = Annotated[Union[NoulQuestion, ChoiceQuestion, ScoreQuestion], Field(discriminator="type")]


class SystemOneRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")     # the SDK's `extra_body` may merge unknown top-level keys
    state: Content | None
    model: str
    questions: Annotated[dict[str, Question], Field(min_length=1)]


class ContractError(Exception):
    """A request the schema accepts but the service refuses. Rendered as {"detail": {"error_type", "message"}}."""

    def __init__(self, status: int, error_type: str, message: str):
        super().__init__(message)
        self.status, self.error_type, self.message = status, error_type, message

    def body(self) -> dict:
        return {"detail": {"error_type": self.error_type, "message": self.message}}


def check_limits(req: SystemOneRequest) -> None:
    """Cardinality limits (the token limits need the tokenizer and are checked by the engine)."""
    for qid, q in req.questions.items():
        if isinstance(q, ChoiceQuestion):
            if len(q.criteria) == 0:
                raise ContractError(400, "invalid_request",
                                    f"Question `{qid}` has no options: a choice needs at least two options in criteria.")
            if len(q.criteria) > MAX_CHOICE_OPTIONS:
                raise ContractError(400, "invalid_request",
                                    f"Too many choices. Must have at most {MAX_CHOICE_OPTIONS} choices (options per choice).")
        elif isinstance(q, ScoreQuestion) and len(q.criteria) > MAX_SCORE_LEVELS:
            raise ContractError(400, "invalid_request",
                                f"Too many score levels in `{qid}`: a score takes 2 to {MAX_SCORE_LEVELS} levels.")


def token_limit_error(longest_branch: int, total: int) -> ContractError | None:
    if longest_branch > MAX_BRANCH_TOKENS:
        return ContractError(400, "max_tokens_exceeded",
                             f"State plus the longest question is {longest_branch} tokens, which exceeds the maximum context "
                             f"length of {MAX_BRANCH_TOKENS} tokens per question.")
    if total > MAX_REQUEST_TOKENS:
        return ContractError(400, "max_tokens_exceeded",
                             f"Request is {total} tokens, which exceeds the maximum context length of {MAX_REQUEST_TOKENS} "
                             f"tokens per request.")
    return None


AUTH_MISSING = {"detail": {"error_type": "authentication_error", "message": "Must supply an API key! Check your request and try again."}}
AUTH_INVALID = {"detail": {"error_type": "authentication_error",
                           "message": "Cannot authenticate with the server. Please check your API key and try again."}}
