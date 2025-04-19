from fastapi import APIRouter
from pydantic import BaseModel
from app.agents.text_agent import NormalAgent

router = APIRouter(prefix="/test/agent", tags=["Agent 测试"])


class AgentRequest(BaseModel):
    query: str


agent = NormalAgent()


@router.post("/run")
async def run_agent(request: AgentRequest):
    result = await agent.run(request.query)
    return {"response": result}
