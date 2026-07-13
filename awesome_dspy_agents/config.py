# pyright: reportMissingTypeStubs=false
import os
from pathlib import Path
from typing import Any, Literal

import dspy
from pydantic import BaseModel, ConfigDict, Field, ValidationError


class LMSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: str = Field(description="Provider namespace for DSPy LM, e.g., 'openai'")
    model: str = Field(description="Model name, e.g., 'gpt-4o-mini'")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    max_tokens: int | None = Field(default=None, ge=1)
    api_base: str | None = None
    api_key: str | None = None
    api_key_env: str | None = Field(
        default=None, description="Env var name for API key"
    )

    def resolve_api_key(self) -> str | None:
        if self.api_key:
            return self.api_key
        if self.api_key_env:
            return os.getenv(self.api_key_env)
        return None


class AgentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    persona: str = Field(
        default="", description="Persona prompt that shapes agent behavior"
    )
    lm: LMSettings | None = None
    module_type: Literal["predict", "chain_of_thought", "react"] = Field(
        default="predict", description="Which DSPy module to use for this agent"
    )
    tools: list[str] = Field(
        default_factory=list, description="Names of tools from the catalog for ReAct"
    )


class JudgeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    discriminative_lm: LMSettings | None = None
    extractive_lm: LMSettings | None = None
    module_type: Literal["predict", "chain_of_thought", "react"] = Field(
        default="predict", description="Judge mode for discriminative/extractive"
    )
    tools: list[str] = Field(
        default_factory=list,
        description="Names of tools from the catalog for Judge ReAct",
    )


class DebateConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    protocol: Literal["classic_adversarial", "consensus_free"] = "classic_adversarial"
    agent_count: int = Field(default=2, ge=2)
    perspectives: list[str] | None = None
    max_iterations: int = Field(default=3, ge=1)
    debate_level: int = Field(default=2, ge=0, le=3)
    adaptive_break: bool = True


class AbsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_iterations: int = Field(default=2, ge=1)
    early_exit: bool = True


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    default_lm: LMSettings | None = None
    agents: dict[str, AgentConfig] = Field(default_factory=dict)
    judge: JudgeConfig = Field(default_factory=JudgeConfig)
    debate: DebateConfig = Field(default_factory=DebateConfig)
    abs: AbsConfig = Field(default_factory=AbsConfig)


def build_lm(settings: LMSettings) -> dspy.LM:
    full_model_id = f"{settings.provider}/{settings.model}"
    api_key = settings.resolve_api_key()
    lm_kwargs: dict[str, Any] = {
        "temperature": settings.temperature,
        "top_p": settings.top_p,
    }
    if settings.max_tokens is not None:
        lm_kwargs["max_tokens"] = settings.max_tokens
    if settings.api_base:
        lm_kwargs["api_base"] = settings.api_base
    if api_key:
        lm_kwargs["api_key"] = api_key
    return dspy.LM(full_model_id, **lm_kwargs)


# YAML loader kept local to avoid hard dep outside CLI usage
def _expand_env_vars_in_data(value: Any) -> Any:
    """Recursively expand environment variables like ${VAR} in strings."""
    if isinstance(value, dict):
        return {k: _expand_env_vars_in_data(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env_vars_in_data(v) for v in value]
    if isinstance(value, str):
        return os.path.expandvars(value)
    return value


def _read_yaml(path: str) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "PyYAML is required to load YAML configs. Install with 'pip install pyyaml'."
        ) from e

    with open(path, encoding="utf-8") as f:
        data: dict[str, Any] = yaml.safe_load(f) or {}
        return _expand_env_vars_in_data(data)


def merge_config_layers(
    base: dict[str, Any], overrides: dict[str, Any]
) -> dict[str, Any]:
    """Recursively merge mappings; lists and scalar values replace."""
    merged = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_config_layers(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(
    path: str,
    overrides: dict[str, Any] | None = None,
    *,
    base_path: str | None = None,
) -> AppConfig:
    data = _read_yaml(path)
    if base_path is not None and Path(base_path).resolve() != Path(path).resolve():
        data = merge_config_layers(_read_yaml(base_path), data)
    if overrides:
        data = merge_config_layers(data, overrides)
    try:
        return AppConfig.model_validate(data)
    except ValidationError as ve:
        raise RuntimeError(f"Invalid configuration file: {ve}") from ve
