from fastapi import FastAPI
from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from app.utils.response import error
from app.routers import chat, user
from app.core.logger import logger

# from app.core.startup import initialize_product_docs
from app.core.config import settings
from fastapi.responses import HTMLResponse

# from contextlib import asynccontextmanager
from app.routers import rag_test
from app.routers import agent_test
from app.routers import eval

logger.info("🚀 应用启动成功")


# @asynccontextmanager  # 异步上下文管理器装饰器
# async def lifespan(app: FastAPI):
#     """应用生命周期管理器

#     参数:
#         app: FastAPI 实例

#     说明:
#         - 启动时按配置决定是否自动向量化产品文档
#         - yield 前执行启动逻辑，yield 后可扩展关闭逻辑
#     """
#     if settings.AUTO_INITIALIZE_DOCS is True:  # 检查是否启用自动初始化配置
#         print(
#             "AUTO_INITIALIZE_DOCS =",
#             settings.AUTO_INITIALIZE_DOCS,
#             type(settings.AUTO_INITIALIZE_DOCS),
#         )
#         logger.info("📦 自动向量化产品说明文档启动中...")
#         await initialize_product_docs()  # 异步初始化产品文档
#     else:
#         logger.info("🚫 已关闭启动时向量化产品说明文档")
#     yield  # 分隔启动和关闭逻辑


app = FastAPI()


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """HTTP异常处理器

    Args:
        request: 请求对象
        exc: Starlette HTTP异常对象

    Returns:
        JSONResponse: 包含错误信息的标准响应

    Note:
        - 处理所有Starlette抛出的HTTP异常
        - 保持原始异常的状态码和错误信息
        - 使用统一的错误响应格式
    """
    return JSONResponse(
        status_code=exc.status_code,  # 保持原始异常状态码
        content=error(message=exc.detail, code=exc.status_code),  # 使用标准错误格式
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """请求参数验证异常处理器

    Args:
        request: 请求对象
        exc: FastAPI请求验证异常对象

    Returns:
        JSONResponse: 返回422状态码的标准错误响应

    Note:
        - 专门处理请求参数验证失败的情况
        - 返回422 Unprocessable Entity状态码
        - 错误信息包含详细的验证错误详情
    """
    return JSONResponse(
        status_code=422,
        content=error(message=str(exc), code=422),  # 将验证错误转为字符串格式
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """全局异常处理器

    Args:
        request: 请求对象
        exc: 捕获的异常对象

    Returns:
        JSONResponse: 返回500状态码的标准错误响应

    Note:
        - 捕获所有未被前面处理器处理的异常
        - 返回统一的500服务器错误响应
        - 避免泄露敏感错误信息到客户端
    """
    return JSONResponse(
        status_code=500,
        content=error(message="服务器内部错误", code=500),  # 使用标准错误格式
    )


app.include_router(user.router)
app.include_router(chat.router)
app.include_router(rag_test.router, prefix="/test")
app.include_router(agent_test.router)
app.include_router(eval.router)


@app.get("/")
def health_check():
    """健康检查端点

    Returns:
        dict: 包含应用状态的基本响应

    Note:
        - 用于Kubernetes/容器健康检查
        - 返回200状态码表示服务正常
        - 简单响应减少带宽消耗
    """
    return {"status": "ok"}  # 返回标准健康状态响应


@app.get("/ws-docs", response_class=HTMLResponse)
async def websocket_docs():
    return """
    <h2>WebSocket 接口说明</h2>
    <p>连接地址: <code>ws://yourdomain.com/ws/chat</code></p>
    <p>发送格式: JSON，例如 <code>{"username": "张三", "message": "你好"}</code></p>
    <p>响应格式: JSON，例如 <code>{"from": "系统", "message": "欢迎你"}</code></p>
    """
