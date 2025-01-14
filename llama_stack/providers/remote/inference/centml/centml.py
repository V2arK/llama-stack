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
    CompletionResponse,
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
    chat_completion_request_to_prompt,
    completion_request_to_prompt,
    content_has_media,
    interleaved_content_as_str,
    request_has_media,
)

from .config import CentMLImplConfig

#
# Example model aliases that map from CentML’s
# published model identifiers to llama-stack's `CoreModelId`.
# Adjust or expand this list based on actual models
# you have available via CentML.
#
MODEL_ALIASES = [
    build_model_alias(
        "meta-llama/Llama-3.2-3B-Instruct-Turbo",
        CoreModelId.llama3_2_3b_instruct.value,
    ),
    # Add any additional aliases as needed
]


class CentMLInferenceAdapter(
    ModelRegistryHelper, Inference, NeedsRequestProviderData
):
    """
    Adapter to use CentML's serverless inference endpoints,
    which adhere to the OpenAI API spec, inside llama-stack.
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
        Creates an OpenAI-compatible client pointing to CentML's
        base URL, using the user's CentML API key.
        """
        api_key = self._get_api_key()
        return OpenAI(api_key=api_key, base_url="https://api.centml.com/openai/v1")

    #
    # COMPLETIONS
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
        This method is called by llama-stack for "completion" style requests
        (non-chat) and must return an async generator.
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
    ) -> CompletionResponse:
        params = await self._get_params(request)
        # CentML (OpenAI-compatible) with synchronous call:
        # Here we wrap it in an async method.
        response = self._get_client().completions.create(**params)
        return process_completion_response(response, self.formatter)

    async def _stream_completion(self, request: CompletionRequest) -> AsyncGenerator:
        params = await self._get_params(request)

        # For streaming, we wrap the streaming generator from
        # CentML/OpenAI in an async generator for llama-stack.
        async def _to_async_generator():
            # The typical openai-python library for streaming:
            #   for chunk in openai.Completion.create(**params, stream=True):
            #       yield chunk
            #
            # We replicate that pattern here:
            stream = self._get_client().completions.create(**params)
            for chunk in stream:
                yield chunk

        stream = _to_async_generator()
        async for chunk in process_completion_stream_response(stream, self.formatter):
            yield chunk

    #
    # CHAT COMPLETIONS
    #

    async def chat_completion(
        self,
        model_id: str,
        messages: List[Message],
        sampling_params: Optional[SamplingParams] = SamplingParams(),
        tools: Optional[List[ToolDefinition]] = None,
        tool_choice: Optional[ToolChoice] = ToolChoice.auto,
        tool_prompt_format: Optional[ToolPromptFormat] = ToolPromptFormat.json,
        response_format: Optional[ResponseFormat] = None,
        stream: Optional[bool] = False,
        logprobs: Optional[LogProbConfig] = None,
    ) -> AsyncGenerator:
        """
        This method is called by llama-stack for "chat completion" style requests.
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
        if "messages" in params:
            response = self._get_client().chat.completions.create(**params)
        else:
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

    def _build_options(
        self,
        sampling_params: Optional[SamplingParams],
        fmt: Optional[ResponseFormat],
    ) -> dict:
        """
        Build the request parameters (temperature, max_tokens, etc.)
        matching the standard OpenAI-based arguments.
        """
        options = get_sampling_options(sampling_params)
        # Provide defaults that might suit your environment
        options.setdefault("max_tokens", 512)

        if fmt:
            if fmt.type == ResponseFormatType.json_schema.value:
                options["response_format"] = {
                    "type": "json_object",
                    "schema": fmt.json_schema,
                }
            elif fmt.type == ResponseFormatType.grammar.value:
                # Example BNF-based grammar
                options["response_format"] = {
                    "type": "grammar",
                    "grammar": fmt.bnf,
                }
            else:
                raise ValueError(f"Unknown response format {fmt.type}")

        return options

    async def _get_params(
        self, request: Union[ChatCompletionRequest, CompletionRequest]
    ) -> dict:
        """
        Converts the llama-stack request object into the parameters
        needed by an OpenAI/CentML call.
        """
        input_dict = {}
        media_present = request_has_media(request)

        if isinstance(request, ChatCompletionRequest):
            # If there's media, we convert the individual messages
            if media_present:
                input_dict["messages"] = [
                    await convert_message_to_openai_dict(m) for m in request.messages
                ]
            else:
                # Otherwise, we convert the entire conversation into a single prompt
                input_dict["prompt"] = await chat_completion_request_to_prompt(
                    request, self.get_llama_model(request.model), self.formatter
                )
        else:
            # CentML currently does not support media in completions
            assert (
                not media_present
            ), "CentML does not support media for Completion requests"
            input_dict["prompt"] = await completion_request_to_prompt(
                request, self.formatter
            )

        # Combine with sampling/response format
        return {
            "model": request.model,
            **input_dict,
            "stream": request.stream,
            **self._build_options(request.sampling_params, request.response_format),
        }

    #
    # EMBEDDINGS
    #

    async def embeddings(
        self,
        model_id: str,
        contents: List[InterleavedContent],
    ) -> EmbeddingsResponse:
        """
        Create embeddings for the requested content using CentML
        (OpenAI-compatible) embeddings endpoint.
        """
        model = await self.model_store.get_model(model_id)
        assert all(
            not content_has_media(content) for content in contents
        ), "CentML does not support media for embeddings"

        response = self._get_client().embeddings.create(
            model=model.provider_resource_id,
            input=[interleaved_content_as_str(content) for content in contents],
        )

        # The openai-compatible embeddings response has a .data list
        embeddings = [item.embedding for item in response.data]
        return EmbeddingsResponse(embeddings=embeddings)
