import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[3]))
import asyncio
import json
from app.llm.openai_client import OpenAILLMClient
from app.llm.LLMModelConfig import LLMModelConfig
from app.rag.embedder import get_aliyun_embedder
import numpy as np


DIALOGUE_SYSTEM_PROMPT = (
    "你是一个昵称叫“通伴”的拟人化助手，风格温和自然，像朋友一样回应用户的情绪与困扰。\n\n"
    "你的目标是：安抚情绪、提供真实具体建议、传递陪伴与支持，表达不冗长。\n\n"
    "请遵循以下沟通原则：\n"
    "1. 用真诚自然的语气，避免说教、模板化；\n"
    "2. 回复控制在 2~4 句，结构建议为“共情 → 建议 → 温和收尾”；\n"
    "3. 避免使用“我是AI”或“我理解”等主语句式开头；\n"
    "4. 内容真实、克制，不跑题、不重复。"
)

DIALOGUE_USER_PROMPT = (
    "你的任务是作为昵称叫“通伴”的拟人化智能助手，与用户进行交流。核心目标是安抚用户的情绪，给予真实、具体又不过度冗长的建议，并让用户感受到被陪伴与支持。\n"
    "以下是用户输入的文本：\n"
    "<用户文本>{{USER_TEXT}}</用户文本>\n"
    "请你敏锐感知用户的情绪，然后按照以下沟通准则输出回应：\n"
    "1. 回复语气自然、真诚，富有情感温度；避免说教式或程式化表达。\n"
    "2. 每条回复控制在2 - 4句内。\n"
    "3. 鼓励使用生活化表达，但避免固定套话；让每一条回复有细微差异，富有个性。\n"
    "4. 禁止使用“我是AI”或“作为AI助手”等自我表述。\n"
    "5. 保证内容围绕用户原始问题展开，真实、克制，不偏题、不冗长。\n"
    "6. 回复开头避免使用“我理解”“我懂”等主语句式，可直接从用户情绪切入，提升自然感与亲切度。\n"
    "7. 结尾建议温和收尾即可，不需频繁强调“我会陪着你”这类话术，避免重复。\n"
    "请写下你的回复。"
)
SCORING_SYSTEM_PROMPT = {
    "role": "system",
    "content": (
        "你是一个评分专家，擅长从语言风格、内容逻辑和情感表达等方面评估对话的质量。\n"
        "请以给定的参考回复为满分基准，严格按照评分维度对模型生成的回复进行打分。\n"
        "每项评分均以参考回复为标准，模型回复与其越接近，得分应越高。\n"
        "最终输出请遵循指定格式返回 JSON 评分结果。"
    ),
}


def build_eval_prompt(input_text: str, output: str, model_output: str) -> str:
    return f"""
请你以评分专家身份，基于以下上下文内容，对模型回复进行以下四个方面的评分，范围为 0~100 分：

1. 拟人化程度：是否自然亲切、像会安慰人的朋友？
2. 清晰度：语言是否通顺、逻辑是否清晰？
3. 简洁度：是否表达紧凑、不啰嗦？
4. 偏题程度：是否答题，是否与豆包回复保持相近主题？
5. 不要增加解析或额外的信息，只按照 JSON 格式返回评分结果（包含评分和建议）：
{{
  "human_likeness": X,
  "clarity": X,
  "conciseness": X,
  "on_topic": X,
  "advice": "请说明你对模型回复在拟人化、清晰度、简洁度、偏题等方面的整体建议"
}}

用户输入：{input_text}
参考回复：{output}
模型回复：{model_output}

请按如下 JSON 格式返回评分结果（包含评分和建议）：
{{
  "human_likeness": X,
  "clarity": X,
  "conciseness": X,
  "on_topic": X,
  "advice": "请说明你对模型回复在拟人化、清晰度、简洁度、偏题等方面的整体建议"
}}
""".strip()


async def evaluate_one(
    dialogue_client: OpenAILLMClient, scoring_client: OpenAILLMClient, item: dict
) -> dict:
    # 生成模型回复
    system = {
        "role": "system",
        "content": DIALOGUE_SYSTEM_PROMPT,
        # "content": "",
    }
    prompt = {"role": "user", "content": f"用户说：{item['input']}"}
    # final_prompt = DIALOGUE_USER_PROMPT.replace("{{USER_TEXT}}", item["input"])
    # prompt = {"role": "user", "content": final_prompt}
    model_response = await dialogue_client.chat(system, prompt, history=[])
    try:
        model_output = model_response.choices[0].message.content.strip()
    except Exception as e:
        return {
            "input": item["input"],
            "output": item["output"],
            "model_output": "",
            "semantic_similarity": "-",
            "score": {
                "human_likeness": "-",
                "clarity": "-",
                "conciseness": "-",
                "on_topic": "-",
                "total": "-",
            },
            "error": f"model_response error: {str(e)}",
        }
    item["model_output"] = model_output

    # 计算语义相似度
    try:
        embedder = get_aliyun_embedder()
        embeddings = await embedder.embed_texts([item["output"], model_output])
        vec1, vec2 = np.array(embeddings[0]), np.array(embeddings[1])
        cos_sim = float(
            np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        )
        semantic_similarity = round(cos_sim * 100, 2)  # 映射到 0~100 分
    except Exception as e:
        semantic_similarity = "-"

    # 构建评分提示词并调用评分模型
    score_prompt = {
        "role": "user",
        "content": build_eval_prompt(item["input"], item["output"], model_output),
    }
    score_response = await scoring_client.chat(
        SCORING_SYSTEM_PROMPT, score_prompt, history=[]
    )
    try:
        content = score_response.choices[0].message.content.strip()
        scores = json.loads(content)
        advice = scores.pop("advice", "")
        # 保持评分原样（模型直接返回百分制）
        scores = {
            k: round(v, 2) for k, v in scores.items() if isinstance(v, (int, float))
        }
        return {
            "input": item["input"],
            "output": item["output"],
            "model_output": model_output,
            "semantic_similarity": semantic_similarity,
            "score": {
                **scores,
                "total": round(sum(scores.values()) / len(scores), 2),
            },
            "advice": advice,
        }
    except Exception as e:
        return {
            "input": item["input"],
            "output": item["output"],
            "model_output": model_output,
            "semantic_similarity": semantic_similarity,
            "score": {
                "human_likeness": "-",
                "clarity": "-",
                "conciseness": "-",
                "on_topic": "-",
                "total": "-",
            },
            "error": f"score_response error: {str(e)}",
        }


async def run_prompt_eval(
    start: int = 0, end: int | None = None, round_id: int = 1
) -> list[dict]:
    input_path = Path(__file__).parent / "prompt_eval_100.jsonl"
    dialogue_client = OpenAILLMClient(LLMModelConfig.DOUBAO_1_5_LITE)
    scoring_client = OpenAILLMClient(LLMModelConfig.QWEN_PLUS_LATEST)

    results = []
    with input_path.open("r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()]
    data = data[start:end]
    for idx, item in enumerate(data):
        if not item.get("output"):
            continue
        result = await evaluate_one(dialogue_client, scoring_client, item)
        results.append(result)
        print(
            f"✅ [{idx}] 完成评估：{item['input'][:20]}... 总分：{result['score'].get('total', '-')}"
        )

    md_path = Path(__file__).parent / "prompt_eval_result.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write(f"# 📋 评估结果（第 {round_id} 轮，共 {len(results)} 条）\n\n")
        f.write(f"**使用提示词：**\n{DIALOGUE_SYSTEM_PROMPT}\n\n")
        f.write("**模型温度：** 0\n\n")
        f.write(f"##  🖨 模型回复\n\n")
        f.write(
            "| 用户输入 | 模型输出 | 参考输出 | 相似度 | 拟人化（0~100分） | 清晰度（0~100分） | 简洁度（0~100分） | 偏题程度（0~100分） | 总分 | 建议 |\n"
        )
        f.write(
            "|----------|----------|----------|--------|------------------|------------------|------------------|--------------------|------|------|\n"
        )
        for item in results:
            score = item.get("score", {})
            row = "| {input} | {model_output} | {output} | {semantic_similarity} | {human_likeness} | {clarity} | {conciseness} | {on_topic} | {total} | {advice} |\n".format(
                input=item["input"].replace("\n", " "),
                model_output=item.get("model_output", "").replace("\n", " "),
                output=item.get("output", "").replace("\n", " "),
                semantic_similarity=item.get("semantic_similarity", "-"),
                human_likeness=score.get("human_likeness", "-"),
                clarity=score.get("clarity", "-"),
                conciseness=score.get("conciseness", "-"),
                on_topic=score.get("on_topic", "-"),
                total=score.get("total", "-"),
                advice=item.get("advice", "").replace("\n", " "),
            )
            f.write(row)

        # 统计维度平均分
        valid_scores = [
            item["score"]
            for item in results
            if isinstance(item.get("score"), dict) and item["score"].get("total") != "-"
        ]
        if valid_scores:
            avg = lambda key: round(
                sum(
                    float(s.get(key, 0))
                    for s in valid_scores
                    if s.get(key) not in ["-", None]
                )
                / len(valid_scores),
                2,
            )
            f.write(f"\n\n## 📊 数据统计\n\n")
            f.write(f"- 总评估数据量：{len(results)} 条\n")
            f.write(f"- 有效评分数量：{len(valid_scores)} 条\n")
            f.write("\n\n## 📈 平均评分统计\n\n")
            f.write(
                f"- 相似度平均分：{round(sum(float(item['semantic_similarity']) for item in results if isinstance(item['semantic_similarity'], (int, float))) / len(valid_scores), 2)}\n"
            )
            f.write(f"- 拟人化平均分：{avg('human_likeness')}\n")
            f.write(f"- 清晰度平均分：{avg('clarity')}\n")
            f.write(f"- 偏题程度平均分：{avg('on_topic')}\n")
            f.write(f"- 总分平均值：{avg('total')}\n")

    return results
