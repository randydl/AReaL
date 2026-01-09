import datetime
import json
import os
import sys
import time
from pathlib import Path

import json5
from pydantic import BaseModel
from qwen_agent.agents.fncall_agent import FnCallAgent
from qwen_agent.llm.schema import Message
from transformers import PreTrainedTokenizer

from areal.experimental.openai import ArealOpenAI
from areal.utils import logging

try:
    from .prompt import SYSTEM_PROMPT
    from .tool_search import GetAllCircuitSummaries
    from .tool_visit import GetCircuitSpecsByName
except ImportError:  # Fallback when executed directly (no package parent known)
    module_dir = Path(__file__).parent
    if str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))
    from prompt import SYSTEM_PROMPT
    from tool_search import GetAllCircuitSummaries
    from tool_visit import GetCircuitSpecsByName


logger = logging.getLogger("Tongyi-DeepResearch react agent")

OBS_START = "<tool_response>"
OBS_END = "\n</tool_response>"

MAX_LLM_CALL_PER_RUN = int(os.getenv("MAX_LLM_CALL_PER_RUN", 100))


class MultiTurnReactAgent(FnCallAgent):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_tokens_per_turn: int = 10000,
        max_llm_calls_per_run: int = 100,
        max_total_tokens: int = 32768,
    ):
        self.tokenizer = tokenizer
        self.max_tokens_per_turn = max_tokens_per_turn
        self.max_llm_calls_per_run = max_llm_calls_per_run
        self.max_total_tokens = max_total_tokens
        self.max_total_tokens_before_finishing = int(max_total_tokens * 0.8)
        self.tool_class = [GetCircuitSpecsByName(), GetAllCircuitSummaries()]
        self.tool_map = {tool.name: tool for tool in self.tool_class}

    def count_tokens(self, messages):
        message_strs = []
        for msg in messages:
            if isinstance(msg, BaseModel):
                msg = msg.model_dump()
                assert "role" in msg and "content" in msg
            message_strs.append(
                f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
            )
        message_strs.append("<|im_start|>assistant\n")
        prompt_token_ids = self.tokenizer.encode("".join(message_strs))
        return len(prompt_token_ids)

    async def call_server(
        self, client: ArealOpenAI, messages: list[dict], max_attempts: int = 100
    ) -> str:
        attempts = 0
        while attempts < max_attempts:
            try:
                completion = await client.chat.completions.create(
                    messages=messages,
                    temperature=1.0,
                    stop=["\n<tool_response>", "<tool_response>"],
                    max_completion_tokens=self.max_tokens_per_turn,
                )
                message = completion.choices[0].message
                assert message, "Error: LLM response is empty."
                return completion, message
            except RuntimeError as e:
                logger.warning(
                    f"RuntimeError during LLM call_server at attempt {attempts}: {e}"
                )
                continue
        raise RuntimeError(
            f"Failed to get response from LLM after {max_attempts} attempts."
        )

    async def run_agent(
        self, data, client: ArealOpenAI, save_path: str | None = None
    ) -> list[list[Message]]:
        start_time = time.time()
        data["qid"]
        question = data["question"]
        answer = data["answer"]
        self.user_prompt = question
        system_prompt = SYSTEM_PROMPT
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]
        stats = dict(
            turns=0,
            num_search=0,
            num_access=0,
        )
        num_llm_calls_available = self.max_llm_calls_per_run
        completions = []
        round = 0
        while num_llm_calls_available > 0:
            # Check whether time is reached
            if time.time() - start_time > 150 * 60:  # 150 minutes in seconds
                prediction = "No answer found after 2h30mins"
                termination = "No answer found after 2h30mins"
                result = {
                    "question": question,
                    "answer": answer,
                    "messages": messages,
                    "prediction": prediction,
                    "termination": termination,
                }
                return result
            round += 1
            stats["turns"] += 1
            num_llm_calls_available -= 1
            completion, message = await self.call_server(client, messages)
            content = message.content
            completions.append(completion)
            messages.append(message)
            # print('>' * 100)
            # print(stats)
            # print(content)
            # print('<' * 100)
            if "<tool_call>" in content and "</tool_call>" in content:
                tool_call = content.split("<tool_call>")[1].split("</tool_call>")[0]
                try:
                    tool_call = json5.loads(tool_call)
                    tool_name = tool_call["name"]
                    tool_args = tool_call.get("arguments", {})
                    result = await self.custom_call_tool(tool_name, tool_args)
                    if tool_name == "get_all_circuit_summaries":
                        stats["num_search"] += 1
                    elif tool_name == "get_circuit_specs_by_name":
                        stats["num_access"] += 1
                except Exception as e:
                    result = f'Error: {e} Tool call must be a valid json contain a valid "name" and "arguments" field.'
                result = "<tool_response>\n" + result + "\n</tool_response>"
                messages.append({"role": "user", "content": result})
            if "<answer>" in content and "</answer>" in content:
                termination = "answer"
                break
            if num_llm_calls_available <= 0 and "<answer>" not in content:
                messages.append(
                    {
                        "role": "user",
                        "content": "Sorry, the number of llm calls exceeds the limit. You should stop making tool calls and, "
                        "based on all the information above, think again and provide what you consider the most likely answer "
                        "in the following format:<think>your final thinking</think>\n<answer>your answer</answer>",
                    }
                )

            max_tokens = self.max_total_tokens_before_finishing
            token_count = self.count_tokens(messages)
            logger.debug(
                f"QID {data['qid']} Round: {round}, token count: {token_count}"
            )

            if token_count > max_tokens:
                logger.debug(
                    f"QID {data['qid']} Token quantity exceeds the limit: {token_count} > {max_tokens}"
                )
                messages.append(
                    {
                        "role": "user",
                        "content": "You have now reached the maximum context length you can handle. "
                        "You should stop making tool calls and, based on all the information above, "
                        "think again and provide what you consider the most likely answer in the following format:"
                        "<think>your final thinking</think>\n<answer>your answer</answer>",
                    }
                )
                completion, message = await self.call_server(client, messages)
                completions.append(completion)
                content = message.content
                messages.append(message)
                if "<answer>" in content and "</answer>" in content:
                    prediction = content.split("<answer>")[1].split("</answer>")[0]
                    termination = "generate an answer as token limit reached"
                else:
                    prediction = content
                    termination = (
                        "format error: generate an answer as token limit reached"
                    )
                result = {
                    "question": question,
                    "answer": answer,
                    "messages": messages,
                    "prediction": prediction,
                    "termination": termination,
                    "completions": completions,
                    "stats": stats,
                }
                token_count = self.count_tokens(messages)
                if token_count > self.max_total_tokens:
                    logger.warning(
                        f"Warning: total token count {token_count} exceeds the hard limit {self.max_total_tokens}."
                    )
                return result

        if "<answer>" in content:
            prediction = content.split("<answer>")[1].split("</answer>")[0]
            termination = "answer"
        else:
            prediction = "No answer found."
            termination = "answer not found"
            if num_llm_calls_available == 0:
                termination = "exceed available llm calls"
        result = {
            "question": question,
            "answer": answer,
            "messages": [
                m.model_dump() if isinstance(m, BaseModel) else m for m in messages
            ],
            "prediction": prediction,
            "termination": termination,
            "completions": completions,  # final completion
            "stats": stats,
        }
        if save_path:
            to_dump = dict(**result)
            to_dump.pop("completions")
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(to_dump, f, ensure_ascii=False, indent=4)
            logger.debug(f"Result dumped to {save_path}")
        return result

    async def custom_call_tool(self, tool_name: str, tool_args: dict, **kwargs):
        if tool_name in self.tool_map:
            raw_result = self.tool_map[tool_name].call(**tool_args)
            result = raw_result
            return result
        else:
            return f"Error: Tool {tool_name} not found"

    async def calc_reward_with_llm_judge(
        self,
        result: dict[str, str],
    ):
        pred_answer = result["prediction"].strip()
        ground_truth = result["answer"].strip()
        if pred_answer == ground_truth:
            reward = 1.0
            print('>' * 100)
            print(f"reward: {reward}")
            print('<' * 100)
        else:
            reward = 0.0
        return reward

    async def make_trajectory(
        self,
        data: dict[str, str],
        client: ArealOpenAI,
        save_path: str | None = None,
    ) -> dict:
        result = await self.run_agent(data, client, save_path=save_path)
        reward = await self.calc_reward_with_llm_judge(result)
        completions = result["completions"]
        last_completion = completions[-1]
        client.set_reward(last_completion.id, reward)
        stats = result["stats"]
        return stats
