# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import warnings
from typing import Annotated, Any, Optional, Union

from pydantic import BaseModel, field_validator

from ....doc_utils import export_module
from ....import_utils import optional_import_block, require_optional_import
from ....llm_config import LLMConfig
from ... import Depends, Tool
from ...dependency_injection import on

with optional_import_block():
    from browser_use import Agent, Browser, BrowserConfig, BrowserContextConfig, BrowserSession, BrowserProfile

    from ....interop.langchain.langchain_chat_model_factory import LangChainChatModelFactory


__all__ = ["BrowserUseResult", "BrowserUseTool", "ExtractedContent"]


@export_module("autogen.tools.experimental.browser_use")
class ExtractedContent(BaseModel):
    """Extracted content from the browser.

    Attributes:
        content: The extracted content.
        url: The URL of the extracted content
    """

    content: str
    url: Optional[str]

    @field_validator("url")
    @classmethod
    def check_url(cls, v: str) -> Optional[str]:
        """Check if the URL is about:blank and return None if it is.

        Args:
            v: The URL to check.
        """
        if v == "about:blank":
            return None
        return v


@export_module("autogen.tools.experimental.browser_use")
class BrowserUseResult(BaseModel):
    """The result of using the browser to perform a task.

    Attributes:
        extracted_content: List of extracted content.
        final_result: The final result.
    """

    extracted_content: list[ExtractedContent]
    final_result: Optional[str]


@require_optional_import(
    [
        "langchain_anthropic",
        "langchain_google_genai",
        "langchain_ollama",
        "langchain_openai",
        "langchain_core",
        "browser_use",
    ],
    "browser-use",
)
@export_module("autogen.tools.experimental")
class BrowserUseTool(Tool):
    """BrowserUseTool is a tool that uses the browser to perform a task."""

    def __init__(  # type: ignore[no-any-unimported]
        self,
        *,
        llm_config: Union[LLMConfig, dict[str, Any]],
        browser_session: Optional[Any] = None,
        agent_kwargs: Optional[dict[str, Any]] = None,
        browser_profile: Optional[dict[str, Any]] = None,
        use_vision: bool = True,
        browser: Optional[Any] = None,  # For backward compatibility
        browser_config: Optional[dict[str, Any]] = None,  # For backward compatibility
    ):
        """Use the browser to perform a task.

        Args:
            llm_config: The LLM configuration.
            browser_session: The browser session to use. If None, a new one will be created.
            agent_kwargs: Additional keyword arguments to pass to the Agent.
            browser_profile: The browser profile configuration to use.
            use_vision: Whether to use vision capabilities. Default is True.
            browser: (Deprecated) For backward compatibility only.
            browser_config: (Deprecated) For backward compatibility only.
        """
        if agent_kwargs is None:
            agent_kwargs = {}

        if browser_profile is None:
            browser_profile = {}

        # Handle backward compatibility
        if browser is not None or browser_config is not None:
            import warnings
            warnings.warn(
                "Parameters 'browser' and 'browser_config' are deprecated. "
                "Use 'browser_session' and 'browser_profile' instead.",
                DeprecationWarning,
                stacklevel=2
            )

        async def browser_use(  # type: ignore[no-any-unimported]
            task: Annotated[str, "The task to perform."],
            llm_config: Annotated[Union[LLMConfig, dict[str, Any]], Depends(on(llm_config))],
            browser_session: Annotated[Any, Depends(on(browser_session))],
            agent_kwargs: Annotated[dict[str, Any], Depends(on(agent_kwargs))],
            browser_profile: Annotated[dict[str, Any], Depends(on(browser_profile))],
            use_vision: Annotated[bool, Depends(on(use_vision))],
        ) -> BrowserUseResult:
            agent_kwargs = agent_kwargs.copy()

            # Create browser_session if not provided
            if browser_session is None:
                browser_profile_obj = BrowserProfile(
                    headless=browser_profile.pop("headless", True),
                    **browser_profile
                )
                browser_session = BrowserSession(browser_profile=browser_profile_obj)

            # set default value for generate_gif
            if "generate_gif" not in agent_kwargs:
                agent_kwargs["generate_gif"] = False

            llm = LangChainChatModelFactory.create_base_chat_model(llm_config)

            max_steps = agent_kwargs.pop("max_steps", 100)
            validate_output = agent_kwargs.pop("validate_output", False)

            agent = Agent(
                task=task,
                llm=llm,
                browser_session=browser_session,
                use_vision=use_vision,
                validate_output=validate_output,
                **agent_kwargs,
            )

            result = await agent.run(max_steps=max_steps)

            extracted_content = [
                ExtractedContent(content=content, url=url)
                for content, url in zip(result.extracted_content(), result.urls())
            ]
            return BrowserUseResult(
                extracted_content=extracted_content,
                final_result=result.final_result(),
            )

        super().__init__(
            name="browser_use",
            description="Use the browser to perform a task.",
            func_or_tool=browser_use,
        )

    @staticmethod
    def _get_controller(llm_config: Union[LLMConfig, dict[str, Any]]) -> Any:
        """This method is kept for backward compatibility but is no longer used."""
        return None
