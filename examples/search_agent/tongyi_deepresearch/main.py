import json
import pandas as pd
from tqdm.auto import tqdm
from openai import OpenAI
from react_agent import ReactInferenceAgent
from joblib import Parallel, delayed


class SingleModelOpenAIClient:
    def __init__(self, client, model):
        self._client = client
        self._model = model

    @property
    def chat(self):
        return self

    @property
    def completions(self):
        return self

    def create(self, **kwargs):
        return self._client.chat.completions.create(
            model=self._model,
            **kwargs,
        )


client = SingleModelOpenAIClient(
    client=OpenAI(
        api_key="EMPTY",
        base_url="http://10.239.2.27:18001/v1",
    ),
    model="Qwen3-8B",
)

agent = ReactInferenceAgent(client)

df = pd.read_json("validation.jsonl", lines=True)


def process_one(idx):
    """处理单条样本（joblib worker）"""
    try:
        question = df.loc[idx, "question"]
        answer = df.loc[idx, "answer"]

        prediction, messages = agent.run(question)

        return idx, prediction, json.dumps(messages, ensure_ascii=False), None
    except Exception as e:
        return idx, None, None, str(e)


# =========================
# joblib 并发
# =========================
N_JOBS = 8  # 根据 vLLM 吞吐调节
BATCH_SIZE = 1  # I/O bound 建议 1

results = Parallel(
    n_jobs=N_JOBS,
    backend="threading",
    batch_size=BATCH_SIZE,
    verbose=10
)(
    delayed(process_one)(idx)
    for idx in tqdm(df.index)
)

# =========================
# 回写结果
# =========================
for idx, prediction, messages, error in results:
    df.loc[idx, "prediction"] = prediction
    df.loc[idx, "messages"] = messages
    if error is not None:
        df.loc[idx, "error"] = error

df.to_json('Qwen3-8B.jsonl', orient='records', force_ascii=False, lines=True)
