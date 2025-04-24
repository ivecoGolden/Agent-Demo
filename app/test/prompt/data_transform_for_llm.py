import json


def convert_prompt_dataset(
    input_file: str = "app/test/prompt/prompt_eval_100.jsonl",
    output_file: str = "app/test/prompt/prompt_eval_messages_100.jsonl",
) -> str:
    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    results = []
    for line in lines:
        data = json.loads(line.strip())
        input_text = data["input"]
        output_text = data["output"]
        message_block = {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "你是一个拟人化的智能助手，昵称叫“通伴”。你的语言风格应温和、真诚，像朋友一样陪伴用户，回应他们的情绪与困扰。\n\n"
                        "你的目标是在以下四个方面做到极致（尽量达到 95 分以上）：\n"
                        "1. 拟人化程度（是否像一个自然、亲切、懂情绪的朋友）\n"
                        "2. 清晰度（语言是否通顺、逻辑自然）\n"
                        "3. 简洁度（表达是否紧凑，是否避免啰嗦、重复）\n"
                        "4. 聚焦度（是否准确回应用户问题，不偏题）\n\n"
                        "请遵循以下沟通准则：\n"
                        "- 回复应体现出共情、理解、安慰，风格自然亲切，避免模板式表达；\n"
                        "- 回复结构建议为“先共情 → 再建议 → 最后鼓励或陪伴式收尾”；\n"
                        "- 每条回复建议 2~4 句，语言应具体、克制、有温度；\n"
                        "- 禁止自称 AI，不得使用“我是AI”或“作为AI助手”等表述；\n"
                        "- 不允许脱离用户情绪主题展开，不引入额外无关话题。"
                    ),
                },
                {"role": "user", "content": input_text},
                {"role": "assistant", "content": output_text},
            ]
        }
        results.append(message_block)

    with open(output_file, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    return f"✅ 转换完成，已保存到 {output_file}"
