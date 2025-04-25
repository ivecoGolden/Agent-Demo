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
    "# CONTEXT #\n"
    "你是“通伴”，一位中文情绪陪伴助手，语气亲切自然，像朋友一样回应用户。\n\n"
    "# OBJECTIVE #\n"
    "你的目标是：\n"
    "1. 真诚回应用户情绪，传达理解与共情；\n"
    "2. 提供贴合问题的具体建议，围绕用户烦恼展开；\n"
    "3. 用轻柔方式引导用户表达更多想法。\n\n"
    "# STYLE #\n"
    "避免书面腔和重复语。\n\n"
    "# TONE #\n"
    "语气温和、关怀、有陪伴感。\n"
    "不得使用 AI 自称，不使用“我会在”“我是 AI”等表述。\n\n"
    "# AUDIENCE #\n"
    "用户可能正处于焦虑、压力、孤独等情绪中，期望得到理解、实用建议和继续对话的空间。\n\n"
    "# RESPONSE #\n"
    "比正常回复精简一半\n"
    "包括：\n"
    "① 情感共鸣；\n"
    "② 具体建议（贴合场景）；\n"
    "③ 自然地提问或回应。\n"
    "禁止使用 Emoji 表情。"
)


def build_scoring_system_prompt(
    input_text: str, output: str, model_output: str
) -> dict:
    return {
        "role": "system",
        "content": (
            "你是一位语言模型输出质量评分专家，具备严谨的判断力和专业的语言分析能力。\n\n"
            "请你基于以下三项内容对模型的生成结果进行评估：\n"
            f"- 用户输入：{input_text}\n"
            f"- 参考回复：{output}\n"
            f"- 模型回复：{model_output}\n\n"
            "你将从四个维度进行 0~100 分打分，并给出一段简明扼要的建议：\n"
            "1. 拟人化程度：回复是否自然亲切，像一个会安慰人的朋友？\n"
            "2. 清晰度：语言是否通顺、有逻辑？\n"
            "3. 简洁度：是否言简意赅、不冗长？\n"
            "4. 偏题程度：回复是否贴合问题，是否与参考回复保持一致主题？\n\n"
            "请仅使用以下 JSON 格式输出评分结果，不要添加额外文字或解析说明：\n"
            "{\n"
            '  "human_likeness": X,\n'
            '  "clarity": X,\n'
            '  "conciseness": X,\n'
            '  "on_topic": X,\n'
            '  "advice": "请简要说明模型在以上四个维度的表现优劣。"\n'
            "}"
        ),
    }


SCORING_USER_PROMPT = (
    "# CONTEXT #\n"
    "你已收到用户输入、参考回复与模型生成结果，系统已提供评分维度说明。\n\n"
    "##############\n\n"
    "# OBJECTIVE #\n"
    "请根据提供内容对模型回复进行评估，输出符合规范的 JSON 格式评分结果。\n\n"
    "##############\n\n"
    "# STYLE #\n"
    "保持语言简练、无附加描述，仅返回 JSON 结果。\n\n"
    "##############\n\n"
    "# TONE #\n"
    "客观、中立、专业。\n\n"
    "##############\n\n"
    "# AUDIENCE #\n"
    "你的评分将用于模型性能评估与提示词优化，目标用户是 AI 工程师。\n\n"
    "##############\n\n"
    "# RESPONSE #\n"
    "请直接返回 JSON 评分，不添加多余解释或格式以外内容。"
)


async def evaluate_one(
    dialogue_client: OpenAILLMClient, scoring_client: OpenAILLMClient, item: dict
) -> dict:
    # 生成模型回复
    system = {
        "role": "system",
        "content": DIALOGUE_SYSTEM_PROMPT,
    }
    prompt = {"role": "user", "content": f"用户输入：{item['input']}"}
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
        "content": SCORING_USER_PROMPT,
    }
    scoring_system_prompt = build_scoring_system_prompt(
        item["input"], item["output"], model_output
    )
    score_response = await scoring_client.chat(
        scoring_system_prompt, score_prompt, history=[]
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
    import random

    random.shuffle(data)
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
        f.write(f"## 🤖 评分模型：{LLMModelConfig.QWEN_PLUS_LATEST.value.model}\n\n")
        f.write("**系统提示词（System Prompt）：**\n\n```text\n")
        scoring_prompt_example = build_scoring_system_prompt(
            input_text="用户输入", output="参考回复", model_output="模型回复"
        )
        f.write(scoring_prompt_example["content"] + "\n")
        f.write("```\n\n")
        f.write("**用户提示词（User Prompt）：**\n\n```text\n")
        f.write(SCORING_USER_PROMPT + "\n")
        f.write("```\n\n")
        f.write(
            f"**模型温度：** {LLMModelConfig.QWEN_PLUS_LATEST.value.temperature}\n\n"
        )
        f.write(f"## 💬 对话模型：{LLMModelConfig.DOUBAO_1_5_LITE.value.model}\n\n")
        f.write("**系统提示词（System Prompt）：**\n\n```text\n")
        f.write(DIALOGUE_SYSTEM_PROMPT + "\n")
        f.write("```\n\n")
        f.write("**用户提示词（User Prompt）：**\n\n```text\n")
        f.write("用户输入：" + "......" + "\n")
        f.write("```\n\n")
        f.write(
            f"**模型温度：** {LLMModelConfig.DOUBAO_1_5_LITE.value.temperature}\n\n"
        )
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
