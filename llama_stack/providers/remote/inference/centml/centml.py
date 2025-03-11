# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import AsyncGenerator, List, Optional, Union

from openai import OpenAI

from llama_models.datatypes import CoreModelId
from llama_models.llama3.api.chat_format import ChatFormat
from llama_models.llama3.api.tokenizer import Tokenizer

from llama_stack.apis.common.content_types import InterleavedContent
from llama_stack.apis.inference import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    CompletionRequest,
    EmbeddingsResponse,
    Inference,
    LogProbConfig,
    Message,
    ResponseFormat,
    ResponseFormatType,
    SamplingParams,
    ToolChoice,
    ToolDefinition,
    ToolPromptFormat,
)
from llama_stack.distribution.request_headers import NeedsRequestProviderData
from llama_stack.providers.utils.inference.model_registry import (
    build_model_alias,
    ModelRegistryHelper,
)
from llama_stack.providers.utils.inference.openai_compat import (
    convert_message_to_openai_dict,
    get_sampling_options,
    process_chat_completion_response,
    process_chat_completion_stream_response,
    process_completion_response,
    process_completion_stream_response,
)
from llama_stack.providers.utils.inference.prompt_adapter import (
    completion_request_to_prompt,
    content_has_media,
    interleaved_content_as_str,
    request_has_media,
)

from .config import CentMLImplConfig

# Example model aliases that map from CentML’s
# published model identifiers to llama-stack's `CoreModelId`.
MODEL_ALIASES = [
    build_model_alias(
        "meta-llama/Llama-3.3-70B-Instruct",
        CoreModelId.llama3_3_70b_instruct.value,
    ),
    build_model_alias(
        "meta-llama/Llama-3.1-405B-Instruct-FP8",
        CoreModelId.llama3_1_405b_instruct.value,
    ),
]


class CentMLInferenceAdapter(
    ModelRegistryHelper, Inference, NeedsRequestProviderData
):
    """
    Adapter to use CentML's serverless inference endpoints,
    which adhere to the OpenAI chat/completions API spec,
    inside llama-stack.
    """

    def __init__(self, config: CentMLImplConfig) -> None:
        super().__init__(MODEL_ALIASES)
        self.config = config
        self.formatter = ChatFormat(Tokenizer.get_instance())

    async def initialize(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass

    def _get_api_key(self) -> str:
        """
        Obtain the CentML API key either from the adapter config
        or from the dynamic provider data in request headers.
        """
        if self.config.api_key is not None:
            return self.config.api_key.get_secret_value()
        else:
            provider_data = self.get_request_provider_data()
            if provider_data is None or not provider_data.centml_api_key:
                raise ValueError(
                    'Pass CentML API Key in the header X-LlamaStack-ProviderData as { "centml_api_key": "<your-api-key>" }'
                )
            return provider_data.centml_api_key

    def _get_client(self) -> OpenAI:
        """
        Creates an OpenAI-compatible client pointing to CentML's base URL,
        using the user's CentML API key.
        """
        api_key = self._get_api_key()
        return OpenAI(api_key=api_key, base_url=self.config.url)

    #
    # COMPLETION (non-chat)
    #

    async def completion(
        self,
        model_id: str,
        content: InterleavedContent,
        sampling_params: Optional[SamplingParams] = SamplingParams(),
        response_format: Optional[ResponseFormat] = None,
        stream: Optional[bool] = False,
        logprobs: Optional[LogProbConfig] = None,
    ) -> AsyncGenerator:
        """
        For "completion" style requests (non-chat).
        """
        model = await self.model_store.get_model(model_id)
        request = CompletionRequest(
            model=model.provider_resource_id,
            content=content,
            sampling_params=sampling_params,
            response_format=response_format,
            stream=stream,
            logprobs=logprobs,
        )
        if stream:
            return self._stream_completion(request)
        else:
            return await self._nonstream_completion(request)

    async def _nonstream_completion(
        self, request: CompletionRequest
    ) -> ChatCompletionResponse:
        params = await self._get_params(request)
        # Using the older "completions" route for non-chat
        response = self._get_client().completions.create(**params)
        return process_completion_response(response, self.formatter)

    async def _stream_completion(
        self, request: CompletionRequest
    ) -> AsyncGenerator:
        params = await self._get_params(request)

        async def _to_async_generator():
            stream = self._get_client().completions.create(**params)
            for chunk in stream:
                yield chunk

        stream = _to_async_generator()
        async for chunk in process_completion_stream_response(
            stream, self.formatter
        ):
            yield chunk

    #
    # CHAT COMPLETION
    #

    async def chat_completion(
        self,
        model_id: str,
        messages: List[Message],
        sampling_params: Optional[SamplingParams] = SamplingParams(),
        tools: Optional[List[ToolDefinition]] = None,
        tool_choice: Optional[ToolChoice] = ToolChoice.auto,
        tool_prompt_format: Optional[ToolPromptFormat] = None,
        response_format: Optional[ResponseFormat] = None,
        stream: Optional[bool] = False,
        logprobs: Optional[LogProbConfig] = None,
    ) -> AsyncGenerator:
        """
        For "chat completion" style requests.
        """
        model = await self.model_store.get_model(model_id)
        request = ChatCompletionRequest(
            model=model.provider_resource_id,
            messages=messages,
            sampling_params=sampling_params,
            tools=tools or [],
            tool_choice=tool_choice,
            tool_prompt_format=tool_prompt_format,
            response_format=response_format,
            stream=stream,
            logprobs=logprobs,
        )
        if stream:
            return self._stream_chat_completion(request)
        else:
            return await self._nonstream_chat_completion(request)

    async def _nonstream_chat_completion(
        self, request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        params = await self._get_params(request)

        # For chat requests, if "messages" is in params -> .chat.completions
        if "messages" in params:
            response = self._get_client().chat.completions.create(**params)
        else:
            # fallback if we ended up only with "prompt"
            response = self._get_client().completions.create(**params)

        return process_chat_completion_response(response, self.formatter)

    async def _stream_chat_completion(
        self, request: ChatCompletionRequest
    ) -> AsyncGenerator:
        params = await self._get_params(request)

        async def _to_async_generator():
            if "messages" in params:
                stream = self._get_client().chat.completions.create(**params)
            else:
                stream = self._get_client().completions.create(**params)
            for chunk in stream:
                yield chunk

        stream = _to_async_generator()
        async for chunk in process_chat_completion_stream_response(
            stream, self.formatter
        ):
            yield chunk

    #
    # HELPER METHODS
    #

    async def _get_params(
        self, request: Union[ChatCompletionRequest, CompletionRequest]
    ) -> dict:
        """
        Build the 'params' dict that the OpenAI (CentML) client expects.
        For chat requests, we always prefer "messages" so that it calls
        the chat endpoint properly.
        """
        input_dict = {}
        media_present = request_has_media(request)

        if isinstance(request, ChatCompletionRequest):
            # For chat requests, always build "messages" from the user messages
            input_dict["messages"] = [
                await convert_message_to_openai_dict(m)
                for m in request.messages
            ]

        else:
            # Non-chat (CompletionRequest)
            assert not media_present, (
                "CentML does not support media for completions"
            )
            input_dict["prompt"] = await completion_request_to_prompt(
                request, self.formatter
            )

        return {
            "model": request.model,
            **input_dict,
            "stream": request.stream,
            **self._build_options(
                request.sampling_params, request.response_format
            ),
        }

    def _build_options(
        self,
        sampling_params: Optional[SamplingParams],
        fmt: Optional[ResponseFormat],
    ) -> dict:
        """
        Build temperature, max_tokens, top_p, etc., plus any response format data.
        """
        options = get_sampling_options(sampling_params)
        options.setdefault("max_tokens", 512)

        if fmt:
            if fmt.type == ResponseFormatType.json_schema.value:
                options["response_format"] = {
                    "type": "json_object",
                    "schema": fmt.json_schema,
                }
            elif fmt.type == ResponseFormatType.grammar.value:
                raise NotImplementedError(
                    "Grammar response format not supported yet"
                )
            else:
                raise ValueError(f"Unknown response format {fmt.type}")

        return options

    #
    # EMBEDDINGS
    #

    async def embeddings(
        self,
        model_id: str,
        contents: List[InterleavedContent],
    ) -> EmbeddingsResponse:
        model = await self.model_store.get_model(model_id)
        # CentML does not support media
        assert all(not content_has_media(c) for c in contents), (
            "CentML does not support media for embeddings"
        )

        resp = self._get_client().embeddings.create(
            model=model.provider_resource_id,
            input=[interleaved_content_as_str(c) for c in contents],
        )
        embeddings = [item.embedding for item in resp.data]
        return EmbeddingsResponse(embeddings=embeddings)
