from pathlib import Path
from fastapi import APIRouter, Query
from app.test.prompt.prompt_eval_runner import run_prompt_eval
from app.test.prompt.data_transform_for_llm import convert_prompt_dataset

router = APIRouter(prefix="/eval", tags=["评测"])


@router.post("/run", summary="执行提示词评估")
async def run_evaluation(
    start: int = Query(default=0, description="评估起始索引"),
    end: int | None = Query(default=None, description="评估结束索引"),
    round_id: int = Query(default=1, description="评估轮次"),
):
    results = await run_prompt_eval(start=start, end=end, round_id=round_id)
    return {
        "message": f"评估完成，共计 {len(results)} 条。",
        "results": results[:5],  # 返回前 5 条作为预览
    }


@router.post("/convert", summary="转换提示词评估数据集为对话格式")
def run_conversion():

    result = convert_prompt_dataset()
    return {"message": result}
