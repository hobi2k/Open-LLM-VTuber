# config_manager/llm.py
from typing import ClassVar, Literal
from pydantic import BaseModel, Field, model_validator
from .i18n import I18nMixin, Description


class StatelessLLMBaseConfig(I18nMixin):
    """Base configuration for StatelessLLM."""

    # interrupt_method. If the provider supports inserting system prompt anywhere in the chat memory, use "system". Otherwise, use "user".
    interrupt_method: Literal["system", "user"] = Field(
        "user", alias="interrupt_method"
    )
    DESCRIPTIONS: ClassVar[dict[str, Description]] = {
        "interrupt_method": Description(
            en="""The method to use for prompting the interruption signal.
            If the provider supports inserting system prompt anywhere in the chat memory, use "system". 
            Otherwise, use "user". You don't need to change this setting.""",
            zh="""用于表示中断信号的方法(提示词模式)。如果LLM支持在聊天记忆中的任何位置插入系统提示词，请使用“system”。
            否则，请使用“user”。您不需要更改此设置。""",
        ),
    }


class StatelessLLMWithTemplate(StatelessLLMBaseConfig):
    """Configuration for OpenAI-compatible LLM providers."""

    base_url: str = Field(..., alias="base_url")
    llm_api_key: str = Field(..., alias="llm_api_key")
    model: str = Field(..., alias="model")
    organization_id: str | None = Field(None, alias="organization_id")
    project_id: str | None = Field(None, alias="project_id")
    template: str | None = Field(None, alias="template")
    temperature: float = Field(1.0, alias="temperature")

    _OPENAI_COMPATIBLE_DESCRIPTIONS: ClassVar[dict[str, Description]] = {
        "base_url": Description(en="Base URL for the API endpoint", zh="API的URL端点"),
        "llm_api_key": Description(en="API key for authentication", zh="API 认证密钥"),
        "organization_id": Description(
            en="Organization ID for the API (Optional)", zh="组织 ID (可选)"
        ),
        "project_id": Description(
            en="Project ID for the API (Optional)", zh="项目 ID (可选)"
        ),
        "model": Description(en="Name of the LLM model to use", zh="LLM 模型名称"),
        "temperature": Description(
            en="What sampling temperature to use, between 0 and 2.",
            zh="使用的采样温度，介于 0 和 2 之间。",
        ),
    }

    DESCRIPTIONS: ClassVar[dict[str, Description]] = {
        **StatelessLLMBaseConfig.DESCRIPTIONS,
        **_OPENAI_COMPATIBLE_DESCRIPTIONS,
    }


class OpenAICompatibleConfig(StatelessLLMBaseConfig):
    """Configuration for OpenAI-compatible LLM providers."""

    base_url: str = Field(..., alias="base_url")
    llm_api_key: str = Field(..., alias="llm_api_key")
    model: str = Field(..., alias="model")
    organization_id: str | None = Field(None, alias="organization_id")
    project_id: str | None = Field(None, alias="project_id")
    temperature: float = Field(1.0, alias="temperature")

    _OPENAI_COMPATIBLE_DESCRIPTIONS: ClassVar[dict[str, Description]] = {
        "base_url": Description(en="Base URL for the API endpoint", zh="API的URL端点"),
        "llm_api_key": Description(en="API key for authentication", zh="API 认证密钥"),
        "organization_id": Description(
            en="Organization ID for the API (Optional)", zh="组织 ID (可选)"
        ),
        "project_id": Description(
            en="Project ID for the API (Optional)", zh="项目 ID (可选)"
        ),
        "model": Description(en="Name of the LLM model to use", zh="LLM 模型名称"),
        "temperature": Description(
            en="What sampling temperature to use, between 0 and 2.",
            zh="使用的采样温度，介于 0 和 2 之间。",
        ),
    }

    DESCRIPTIONS: ClassVar[dict[str, Description]] = {
        **StatelessLLMBaseConfig.DESCRIPTIONS,
        **_OPENAI_COMPATIBLE_DESCRIPTIONS,
    }


# Ollama config is completely the same as OpenAICompatibleConfig


class OllamaConfig(OpenAICompatibleConfig):
    """Configuration for Ollama API."""

    llm_api_key: str = Field("default_api_key", alias="llm_api_key")
    keep_alive: float = Field(-1, alias="keep_alive")
    unload_at_exit: bool = Field(True, alias="unload_at_exit")
    interrupt_method: Literal["system", "user"] = Field(
        "system", alias="interrupt_method"
    )

    # Ollama-specific descriptions
    _OLLAMA_DESCRIPTIONS: ClassVar[dict[str, Description]] = {
        "llm_api_key": Description(
            en="API key for authentication (defaults to 'default_api_key' for Ollama)",
            zh="API 认证密钥 (Ollama 默认为 'default_api_key')",
        ),
        "keep_alive": Description(
            en="Keep the model loaded for this many seconds after the last request. "
            "Set to -1 to keep the model loaded indefinitely.",
            zh="在最后一个请求之后保持模型加载的秒数。设置为 -1 以无限期保持模型加载。",
        ),
        "unload_at_exit": Description(
            en="Unload the model when the program exits.",
            zh="是否在程序退出时卸载模型。",
        ),
    }

    DESCRIPTIONS: ClassVar[dict[str, Description]] = {
        **OpenAICompatibleConfig.DESCRIPTIONS,
        **_OLLAMA_DESCRIPTIONS,
    }


class OpenCodeConfig(StatelessLLMBaseConfig):
    """Configuration for an OpenCode headless server."""

    executable: str = Field("auto", alias="executable")
    base_url: str = Field("http://127.0.0.1:4096", alias="base_url")
    provider_id: str = Field(..., alias="provider_id")
    model: str = Field(..., alias="model")
    agent: str = Field("vtuber", alias="agent")
    interaction_mode: Literal["character", "coding"] = Field(
        "character", alias="interaction_mode"
    )
    launch_mode: Literal["direct", "omlx"] = Field("direct", alias="launch_mode")
    session_id: str = Field("", alias="session_id")
    new_session_title: str = Field("", max_length=120, alias="new_session_title")
    workspace_directory: str = Field(".", alias="workspace_directory")
    timeout: float = Field(300, gt=0, alias="timeout")
    keep_sessions: bool = Field(False, alias="keep_sessions")
    allow_tools: bool = Field(False, alias="allow_tools")
    permission_mode: Literal["disabled", "manual", "auto", "plan"] | None = Field(
        None, alias="permission_mode"
    )
    show_reasoning: bool = Field(False, alias="show_reasoning")
    server_username: str | None = Field(None, alias="server_username")
    server_password: str | None = Field(None, alias="server_password")

    _OPENCODE_DESCRIPTIONS: ClassVar[dict[str, Description]] = {
        "executable": Description(
            en="OpenCode executable path or 'auto' for PATH discovery",
            zh="OpenCode 可执行文件路径，或使用 'auto' 从 PATH 自动查找",
        ),
        "base_url": Description(
            en="Base URL of the OpenCode headless server",
            zh="OpenCode headless 服务器的基础 URL",
        ),
        "provider_id": Description(
            en="Provider ID configured in OpenCode",
            zh="OpenCode 中配置的模型提供商 ID",
        ),
        "model": Description(
            en="Model ID configured for the selected OpenCode provider",
            zh="所选 OpenCode 提供商中配置的模型 ID",
        ),
        "agent": Description(
            en="OpenCode agent name used for conversations",
            zh="用于对话的 OpenCode 智能体名称",
        ),
        "workspace_directory": Description(
            en="Directory OpenCode uses as the conversation workspace",
            zh="OpenCode 用作对话工作区的目录",
        ),
        "timeout": Description(
            en="Maximum number of seconds to wait for a response",
            zh="等待响应的最长秒数",
        ),
        "keep_sessions": Description(
            en="Keep generated OpenCode sessions for debugging",
            zh="保留生成的 OpenCode 会话以便调试",
        ),
        "allow_tools": Description(
            en="Allow the selected OpenCode agent to use tools",
            zh="允许所选 OpenCode 智能体使用工具",
        ),
        "permission_mode": Description(
            en="Choose disabled, manual approval, automatic approval, or plan mode",
            zh="选择禁用、手动批准、自动批准或计划模式",
        ),
        "server_username": Description(
            en="Username for an authenticated OpenCode server",
            zh="启用认证的 OpenCode 服务器用户名",
        ),
        "server_password": Description(
            en="Password for an authenticated OpenCode server",
            zh="启用认证的 OpenCode 服务器密码",
        ),
    }

    DESCRIPTIONS: ClassVar[dict[str, Description]] = {
        **StatelessLLMBaseConfig.DESCRIPTIONS,
        **_OPENCODE_DESCRIPTIONS,
    }

    @model_validator(mode="after")
    def resolve_permission_mode(self):
        if self.permission_mode is None:
            self.permission_mode = "auto" if self.allow_tools else "disabled"
        self.allow_tools = self.permission_mode != "disabled"
        return self


class CLIAgentConfig(StatelessLLMBaseConfig):
    """Configuration shared by installed one-shot agent CLIs."""

    executable: str = Field("auto", min_length=1, alias="executable")
    launch_mode: Literal["direct", "omlx"] = Field("direct", alias="launch_mode")
    interaction_mode: Literal["character", "coding"] = Field(
        "character", alias="interaction_mode"
    )
    session_id: str = Field("", alias="session_id")
    new_session_title: str = Field("", max_length=120, alias="new_session_title")
    model: str = Field("", alias="model")
    provider: str = Field("", alias="provider")
    workspace_directory: str = Field(".", min_length=1, alias="workspace_directory")
    timeout: float = Field(300, gt=0, alias="timeout")
    show_reasoning: bool = Field(False, alias="show_reasoning")
    reasoning_effort: Literal[
        "default", "none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra"
    ] = Field("default", alias="reasoning_effort")
    allow_tools: bool = Field(False, alias="allow_tools")
    permission_mode: Literal["disabled", "manual", "auto", "plan"] | None = Field(
        None, alias="permission_mode"
    )

    @model_validator(mode="after")
    def resolve_permission_mode(self):
        if self.permission_mode is None:
            self.permission_mode = "auto" if self.allow_tools else "disabled"
        self.allow_tools = self.permission_mode != "disabled"
        return self


class LmStudioConfig(OpenAICompatibleConfig):
    """Configuration for LM Studio."""

    llm_api_key: str = Field("default_api_key", alias="llm_api_key")
    base_url: str = Field("http://localhost:1234/v1", alias="base_url")
    interrupt_method: Literal["system", "user"] = Field(
        "system", alias="interrupt_method"
    )


class OpenAIConfig(OpenAICompatibleConfig):
    """Configuration for Official OpenAI API."""

    base_url: str = Field("https://api.openai.com/v1", alias="base_url")
    interrupt_method: Literal["system", "user"] = Field(
        "system", alias="interrupt_method"
    )


class GeminiConfig(OpenAICompatibleConfig):
    """Configuration for Gemini API."""

    base_url: str = Field(
        "https://generativelanguage.googleapis.com/v1beta/openai/", alias="base_url"
    )
    interrupt_method: Literal["system", "user"] = Field(
        "user", alias="interrupt_method"
    )


class MistralConfig(OpenAICompatibleConfig):
    """Configuration for Mistral API."""

    base_url: str = Field("https://api.mistral.ai/v1", alias="base_url")
    interrupt_method: Literal["system", "user"] = Field(
        "user", alias="interrupt_method"
    )


class ZhipuConfig(OpenAICompatibleConfig):
    """Configuration for Zhipu API."""

    base_url: str = Field("https://open.bigmodel.cn/api/paas/v4/", alias="base_url")


class DeepseekConfig(OpenAICompatibleConfig):
    """Configuration for Deepseek API."""

    base_url: str = Field("https://api.deepseek.com/v1", alias="base_url")


class GroqConfig(OpenAICompatibleConfig):
    """Configuration for Groq API."""

    base_url: str = Field("https://api.groq.com/openai/v1", alias="base_url")
    interrupt_method: Literal["system", "user"] = Field(
        "system", alias="interrupt_method"
    )


class ClaudeConfig(StatelessLLMBaseConfig):
    """Configuration for OpenAI Official API."""

    base_url: str = Field("https://api.anthropic.com", alias="base_url")
    llm_api_key: str = Field(..., alias="llm_api_key")
    model: str = Field(..., alias="model")
    interrupt_method: Literal["system", "user"] = Field(
        "user", alias="interrupt_method"
    )

    _CLAUDE_DESCRIPTIONS: ClassVar[dict[str, Description]] = {
        "base_url": Description(
            en="Base URL for Claude API", zh="Claude API 的API端点"
        ),
        "llm_api_key": Description(en="API key for authentication", zh="API 认证密钥"),
        "model": Description(
            en="Name of the Claude model to use", zh="要使用的 Claude 模型名称"
        ),
    }

    DESCRIPTIONS: ClassVar[dict[str, Description]] = {
        **StatelessLLMBaseConfig.DESCRIPTIONS,
        **_CLAUDE_DESCRIPTIONS,
    }


class LlamaCppConfig(StatelessLLMBaseConfig):
    """Configuration for LlamaCpp."""

    model_path: str = Field(..., alias="model_path")
    interrupt_method: Literal["system", "user"] = Field(
        "system", alias="interrupt_method"
    )

    _LLAMA_DESCRIPTIONS: ClassVar[dict[str, Description]] = {
        "model_path": Description(
            en="Path to the GGUF model file", zh="GGUF 模型文件路径"
        ),
    }

    DESCRIPTIONS: ClassVar[dict[str, Description]] = {
        **StatelessLLMBaseConfig.DESCRIPTIONS,
        **_LLAMA_DESCRIPTIONS,
    }


class StatelessLLMConfigs(I18nMixin, BaseModel):
    """Pool of LLM provider configurations.
    This class contains configurations for different LLM providers."""

    stateless_llm_with_template: StatelessLLMWithTemplate | None = Field(
        None, alias="stateless_llm_with_template"
    )
    openai_compatible_llm: OpenAICompatibleConfig | None = Field(
        None, alias="openai_compatible_llm"
    )
    ollama_llm: OllamaConfig | None = Field(None, alias="ollama_llm")
    opencode_llm: OpenCodeConfig | None = Field(None, alias="opencode_llm")
    claude_code_llm: CLIAgentConfig | None = Field(None, alias="claude_code_llm")
    codex_cli_llm: CLIAgentConfig | None = Field(None, alias="codex_cli_llm")
    hermes_cli_llm: CLIAgentConfig | None = Field(None, alias="hermes_cli_llm")
    lmstudio_llm: LmStudioConfig | None = Field(None, alias="lmstudio_llm")
    openai_llm: OpenAIConfig | None = Field(None, alias="openai_llm")
    gemini_llm: GeminiConfig | None = Field(None, alias="gemini_llm")
    zhipu_llm: ZhipuConfig | None = Field(None, alias="zhipu_llm")
    deepseek_llm: DeepseekConfig | None = Field(None, alias="deepseek_llm")
    groq_llm: GroqConfig | None = Field(None, alias="groq_llm")
    claude_llm: ClaudeConfig | None = Field(None, alias="claude_llm")
    llama_cpp_llm: LlamaCppConfig | None = Field(None, alias="llama_cpp_llm")
    mistral_llm: MistralConfig | None = Field(None, alias="mistral_llm")

    DESCRIPTIONS: ClassVar[dict[str, Description]] = {
        "stateless_llm_with_template": Description(
            en="Stateless LLM with Template", zh=""
        ),
        "openai_compatible_llm": Description(
            en="Configuration for OpenAI-compatible LLM providers",
            zh="OpenAI兼容的语言模型提供者配置",
        ),
        "opencode_llm": Description(
            en="Configuration for an OpenCode headless server",
            zh="OpenCode headless 服务器配置",
        ),
        "claude_code_llm": Description(
            en="Configuration for Claude Code CLI", zh="Claude Code CLI 配置"
        ),
        "codex_cli_llm": Description(
            en="Configuration for Codex CLI", zh="Codex CLI 配置"
        ),
        "hermes_cli_llm": Description(
            en="Configuration for Hermes CLI", zh="Hermes CLI 配置"
        ),
        "ollama_llm": Description(en="Configuration for Ollama", zh="Ollama 配置"),
        "lmstudio_llm": Description(
            en="Configuration for LM Studio", zh="LM Studio 配置"
        ),
        "openai_llm": Description(
            en="Configuration for Official OpenAI API", zh="官方 OpenAI API 配置"
        ),
        "gemini_llm": Description(
            en="Configuration for Gemini API", zh="Gemini API 配置"
        ),
        "mistral_llm": Description(
            en="Configuration for Mistral API", zh="Mistral API 配置"
        ),
        "zhipu_llm": Description(en="Configuration for Zhipu API", zh="Zhipu API 配置"),
        "deepseek_llm": Description(
            en="Configuration for Deepseek API", zh="Deepseek API 配置"
        ),
        "groq_llm": Description(en="Configuration for Groq API", zh="Groq API 配置"),
        "claude_llm": Description(
            en="Configuration for Claude API", zh="Claude API配置"
        ),
        "llama_cpp_llm": Description(
            en="Configuration for local Llama.cpp", zh="本地Llama.cpp配置"
        ),
    }
