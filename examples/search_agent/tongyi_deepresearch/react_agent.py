import json5
import logging
from transformers import AutoTokenizer
from prompt import SYSTEM_PROMPT
from tool_visit import GetCircuitSpecsByName
from tool_search import GetAllCircuitSummaries


logger = logging.getLogger("react-inference-agent")


class ReactInferenceAgent:
    FINAL_ANSWER_PROMPT = (
        "Stop making tool calls and provide your final answer in the format:\n"
        "<think>...</think>\n<answer>...</answer>"
    )

    def __init__(
        self,
        client,
        system_prompt=SYSTEM_PROMPT,
        tools=[GetAllCircuitSummaries(), GetCircuitSpecsByName()],
        max_llm_calls=10,
        max_total_tokens=32768,
    ):
        self.client = client
        self.system_prompt = system_prompt
        self.max_llm_calls = max_llm_calls
        self.max_total_tokens = max_total_tokens
        self.tool_map = {tool.name: tool for tool in tools}
        self.tokenizer = AutoTokenizer.from_pretrained('/nas_train/app.e0016372/models/Qwen3-8B')

    def count_tokens(self, messages):
        message_strs = []
        for msg in messages:
            message_strs.append(
                f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
            )
        message_strs.append("<|im_start|>assistant\n")
        prompt_token_ids = self.tokenizer.encode("".join(message_strs))
        return len(prompt_token_ids)
    # --------------------------------------------------
    # Public entry
    # --------------------------------------------------
    def run(self, user_prompt):
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        for _ in range(self.max_llm_calls):
            message = self._call_llm(messages)
            content = message['content']
            messages.append(message)

            token_count = self.count_tokens(messages)
            if token_count > self.max_total_tokens:
                break

            # 1️⃣ Tool call
            if "<tool_call>" in content:
                tool_result = self._handle_tool_call(content)
                messages.append({
                    "role": "user",
                    "content": f"<tool_response>\n{tool_result}\n</tool_response>",
                })
                continue

            # 2️⃣ Final answer
            if self._has_answer(content):
                answer = self._extract_answer(content)
                return answer, messages

        # 3️⃣ Fallback
        messages.append({
            "role": "user",
            "content": self.FINAL_ANSWER_PROMPT
        })
        message = self._call_llm(messages)
        content = message['content']
        messages.append(message)
        answer = self._extract_answer(content)
        return answer, messages

    # --------------------------------------------------
    # LLM
    # --------------------------------------------------
    def _call_llm(self, messages, max_attempts=5):
        for attempt in range(max_attempts):
            try:
                completion = self.client.chat.completions.create(
                    messages=messages,
                    # temperature=1.0,
                    # max_completion_tokens=self.max_completion_tokens,
                )
                message = completion.choices[0].message
                if not message:
                    raise RuntimeError("Empty LLM response")
                return {'role': 'assistant', 'content': message.content}
            except Exception as e:
                if attempt == max_attempts - 1:
                    raise
                logger.warning(f"LLM call failed, retrying: {e}")

    # --------------------------------------------------
    # Tool
    # --------------------------------------------------
    def _handle_tool_call(self, content):
        try:
            raw = content.split("<tool_call>")[1].split("</tool_call>")[0]
            call = json5.loads(raw)

            name = call["name"]
            args = call.get("arguments", {})

            tool = self.tool_map.get(name)
            if not tool:
                return f"Error: tool `{name}` not found"

            return tool.call(**args)

        except Exception as e:
            return f"Tool call error: {e}"

    # --------------------------------------------------
    # Answer helpers
    # --------------------------------------------------
    def _has_answer(self, content):
        return "<answer>" in content and "</answer>" in content

    def _extract_answer(self, content):
        if not self._has_answer(content):
            return content.strip()
        return content.split("<answer>")[1].split("</answer>")[0].strip()
