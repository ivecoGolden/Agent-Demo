# 🚀 UserApp

一个基于 FastAPI 的用户管理应用，使用 PostgreSQL 存储数据，并通过 Docker 进行部署和管理。

---

## 📦 技术栈

- **FastAPI** - 高性能 Web 框架
- **SQLAlchemy** - ORM 数据库操作
- **PostgreSQL** - 数据持久化
- **Alembic** - 数据库迁移工具
- **Docker + Docker Compose** - 容器化部署
- **Milvus** - 向量数据库，支持高效向量检索
- **Pydantic** - 数据校验
- **JWT** - 身份认证

---

## 🛠 项目结构

- `app/`：应用主代码目录，包含核心逻辑、路由、服务、模型、提示词等
- `alembic/`：数据库迁移脚本，配合 Alembic 使用
- `milvus/`：Milvus 向量数据库的部署与配置文件
- `postgre/`：PostgreSQL 数据库容器配置
- `docs/`：产品文档或说明书
- `tests/`：测试脚本
- `docker-compose.yml` 等：Docker 服务编排文件
- `wait-for-it.sh`：用于容器间启动顺序控制的脚本

---

## ⚙️ 环境变量配置（`.env`）

你可以参考 `.env.example` 文件：

```env
# 应用名称
APP_NAME=UserApp

# 运行环境（如：dev / production）
ENV=dev

# 数据库连接字符串
DATABASE_URL=postgresql://xxx:xxx@db:000/xxx

# JWT 加密用的密钥，请替换为实际部署密钥
SECRET_KEY=your-secret-key-here

# Access Token 过期时间（分钟），示例：1440 = 1天
ACCESS_TOKEN_EXPIRE_MINUTES=1440

# PostgreSQL数据库用户名(容器部署时使用)
POSTGRES_USER=""
# PostgreSQL数据库密码(容器部署时使用)
POSTGRES_PASSWORD=""
# PostgreSQL默认数据库名(容器部署时使用)
POSTGRES_DB=""

# 阿里云大模型API密钥（需替换为实际密钥）
ALI_LLM_KEY=""
# 阿里云大模型API基础URL地址（需替换为实际地址）
ALI_LLM_BASE_URL=""

# 火山引擎大模型API密钥（需替换为实际密钥）
HUOSHAN_LLM_KEY=""
# 火山引擎大模型API基础URL地址（需替换为实际地址）
HUOSHAN_LLM_BASE_URL=""

# LangSmith 链路追踪配置
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=""
LANGSMITH_ENDPOINT="https://api.smith.langchain.com"
LANGSMITH_PROJECT="xxxx"
```

---

## 🚀 本地开发运行

1. 安装依赖

```bash
pip install -r requirements.txt
```

2. 创建 Docker 网络（确保各服务能互通）

```bash
docker network create mg_backend
```

3. 启动数据库服务

```bash
cd postgre
docker-compose up -d
```

4. 启动向量数据库 Milvus

```bash
cd ../milvus
docker-compose up -d
```

5. 启动主服务（包含后端服务等）

```bash
cd ..
docker-compose up -d
```

6. 启动应用（默认监听端口为 7001，可在根目录 docker-compose.yml 中修改）

```bash
uvicorn app.main:app --reload --port 7001
```

访问地址：http://localhost:7001

---

## 🧬 数据库迁移（Alembic）

- 生成迁移脚本

```bash
alembic revision --autogenerate -m "message"
```

- 应用迁移到数据库

```bash
alembic upgrade head
```

---

## 🔐 API 认证

采用 JWT Token 登录验证，默认过期时间为 1 天，可通过 `.env` 修改 `ACCESS_TOKEN_EXPIRE_MINUTES`

---

## 🧪 测试

```bash
pytest -v
```

---

## 📌 注意事项

- `.env` 文件请勿上传到版本库（已加入 `.gitignore`）
- 初次部署数据库请确保数据卷挂载成功
- 建议使用 `wait-for-it.sh` 等工具确保容器间启动顺序(已经废弃)

---

## 🧑‍💻 作者

Made with ❤️ by Yolk

---
