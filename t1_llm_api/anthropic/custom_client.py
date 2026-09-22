import json
import aiohttp
import requests

from commons.models.message import Message
from commons.models.role import Role
from t1_llm_api.base_client import AIClient


class CustomAnthropicAIClient(AIClient):
    """
    Custom HTTP client for Anthropic's Claude API.

    This implementation uses raw HTTP requests (requests/aiohttp) instead of
    the official SDK, demonstrating how to interact with Claude's API directly
    and handle its Server-Sent Events (SSE) streaming format.
    """

    def response(self, messages: list[Message], **kwargs) -> Message:
        """
        Get a synchronous response using raw HTTP POST request.

        Args:
            messages (list[Message]): The conversation history.
            **kwargs: Additional parameters like max_tokens (default: 1024).

        Returns:
            Message: The AI's response message.

        Raises:
            ValueError: If the API response contains no content blocks.
            Exception: If the HTTP request fails (non-200 status code).

        Note:
            Requires 'x-api-key' header and 'anthropic-version' header.
            Claude's API returns content as an array of content blocks.
            The response is printed to stdout before being returned.
        """

        headers = {
            "x-api-key": self._api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }

        body = {
            "model": self._model_name,
            "system": self._system_prompt,
            "max_tokens": 1024,
            "stream": False,
            "messages": [message.to_dict() for message in messages],
        }

        response = requests.post(
            url=self._endpoint + "/v1/messages", headers=headers, json=body
        )

        print(response.status_code)
        print(response.text)

        if response.status_code == 200:
            data = response.json()
            content_blocks = data.get("content", [])

            if content_blocks:
                content = "".join(
                    block.get("text", "")
                    for block in content_blocks
                    if block.get("type") == "text"
                )

                print(f"Assistant: {content}")
                return Message(role=Role.ASSISTANT, content=content)

        raise ValueError("No content blocks present in the response")
        # TODO:
        # https://platform.claude.com/docs/en/build-with-claude/working-with-messages
        # 0. Make a request in Postman to see the request and response
        # 1. Prepare headers dict with:
        #   - "x-api-key" (self api key)
        #   - "Content-Type" ("application/json")
        #   - "anthropic-version" ("2023-06-01")
        # 2. Prepare request data dict:
        #   - "model" (self model_name)
        #   - "system" (self system_prompt)
        #   - "max_tokens" (1024)
        #   - "messages" ([message.to_dict() for message in messages])
        # 3. Execute post request to AI API `requests.post(url=self._endpoint, headers=headers, json=request_data)`
        # 4.1. If response status code is 200 then:
        #   - get response json
        #   - get content block
        #   - if content blocks are present:
        #       - get content: "".join(block.get("text", "") for block in content_blocks if block.get("type") == "text")
        #       - print content
        #       - return ASSISTANT message (role assistant, content is generated content)
        #   - raise ValueError("No content blocks present in the response")
        # 4.2. Otherwise raise Exception(f"HTTP {response.status_code}: {response.text}")

    async def stream_response(
        self,
        messages: list[Message],
        **kwargs,
    ) -> Message:
        headers = {
            "x-api-key": self._api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }

        body = {
            "model": self._model_name,
            "system": self._system_prompt,
            "max_tokens": kwargs.get("max_tokens", 1024),
            "stream": True,
            "messages": [message.to_dict() for message in messages],
        }

        contents = []

        async with aiohttp.ClientSession() as session:
            async with session.post(
                url=self._endpoint + "/v1/messages",
                headers=headers,
                json=body,
            ) as response:

                if response.status == 200:
                    async for line in response.content:
                        line_str = line.decode("utf-8").strip()

                        if not line_str.startswith("data: "):
                            continue

                        data = line_str[6:].strip()

                        if not data:
                            continue

                        try:
                            parsed_data = json.loads(data)
                        except json.JSONDecodeError:
                            continue

                        event_type = parsed_data.get("type")

                        if event_type == "content_block_delta":
                            delta = parsed_data.get("delta", {})

                            if delta.get("type") == "text_delta":
                                text_content = delta.get("text", "")

                                if text_content:
                                    print(text_content, end="")
                                    contents.append(text_content)

                        elif event_type == "message_stop":
                            break

                else:
                    error_text = await response.text()
                    print(f"{response.status} {error_text}")
                    raise Exception(f"HTTP {response.status}: {error_text}")

        print()

        return Message(
            role=Role.ASSISTANT,
            content="".join(contents),
        )
        # TODO:
        # https://platform.claude.com/docs/en/build-with-claude/streaming
        # 0. Make a request in Postman to see the request and response
        # 1. Prepare headers dict with:
        #   - "x-api-key" (self api key)
        #   - "Content-Type" ("application/json")
        #   - "anthropic-version" ("2023-06-01")
        # 2. Prepare request data dict:
        #   - "model" (self model_name)
        #   - "system" (self system_prompt)
        #   - "max_tokens" (kwargs.get("max_tokens", 1024))
        #   - "stream" (True)
        #   - "messages" ([msg.to_dict() for msg in messages])
        # 3. Initialize empty contents list to collect streamed text chunks
        # 4. Create aiohttp ClientSession using `async with aiohttp.ClientSession() as session:`
        # 5. Execute async POST request using `async with session.post(url=self._endpoint, headers=headers, json=request_data) as response:`
        # 6.1. If response status is 200:
        #   - iterate through response content lines using `async for line in response.content:`
        #   - decode each line: `line_str = line.decode('utf-8').strip()`
        #   - check if line starts with "data: " (SSE format)
        #   - extract JSON data: `data = line_str[6:].strip()`
        #   - parse JSON data: `parsed_data = json.loads(data)`
        #   - get event type: `event_type = parsed_data.get("type")`
        #   - if event_type == "content_block_delta":
        #       - get delta: `delta = parsed_data.get("delta", {})`
        #       - if delta.get("type") == "text_delta":
        #           - get text content: `text_content = delta.get("text", "")`
        #           - if text_content is not empty:
        #               - print text_content without newline (end='')
        #               - append text_content to contents list
        #   - else if event_type is `message_stop` then break from the loop
        # 6.2. Otherwise:
        #   - get error text: `error_text = await response.text()`
        #   - print error: f"{response.status} {error_text}"
        # 7. Print empty line (for formatting)
        # 8. Return ASSISTANT message with joined contents: `Message(role=Role.ASSISTANT, content=''.join(contents))`
