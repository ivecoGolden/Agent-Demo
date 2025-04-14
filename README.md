# 🚀 UserApp

一个基于 FastAPI 的用户管理应用，使用 PostgreSQL 存储数据，并通过 Docker 进行部署和管理。

---

## 📦 技术栈

- **FastAPI** - 高性能 Web 框架
- **SQLAlchemy** - ORM 数据库操作
- **PostgreSQL** - 数据持久化
- **Alembic** - 数据库迁移工具
- **Docker + Docker Compose** - 容器化部署
- **Pydantic** - 数据校验
- **JWT** - 身份认证

---

## 🛠 项目结构

```
user_app/
├── app/                    # 应用主逻辑
│   ├── api/                # 路由定义
│   ├── core/               # 设置、配置项
│   ├── models/             # 数据模型
│   ├── schemas/            # Pydantic 模型
│   ├── services/           # 服务层
│   └── main.py             # 应用入口
├── alembic/                # 数据库迁移脚本
├── requirements.txt        # Python依赖列表
├── docker-compose.yml      # 服务编排
├── Dockerfile              # 镜像构建定义
├── .env                    # 环境变量（不提交）
└── README.md               # 项目说明文档
```

---

## ⚙️ 环境变量配置（`.env`）

你可以参考 `.env.example` 文件：

```env
APP_NAME=UserApp
ENV=dev
DATABASE_URL=postgresql://postgres:postgres@db:5432/userdemo
SECRET_KEY=生成你自己的随机密钥
ACCESS_TOKEN_EXPIRE_MINUTES=1440
```

---

## 🚀 本地开发运行

1. 安装依赖

```bash
pip install -r requirements.txt
```

2. 本地数据库运行（可选）

```bash
docker-compose up db
```

3. 启动应用

```bash
uvicorn app.main:app --reload
```

访问地址： http://localhost:8000

---

## 🐳 Docker 部署方式

```bash
docker-compose up --build
```

默认监听地址： http://localhost:9000

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

## 🛰️ CI/CD 推荐（可选）

- 使用 GitHub Actions 自动构建镜像并部署到远程服务器
- 或接入 Jenkins / Drone 自定义流程

---

## 📌 注意事项

- `.env` 文件请勿上传到版本库（已加入 `.gitignore`）
- 初次部署数据库请确保数据卷挂载成功
- 建议使用 `wait-for-it.sh` 等工具确保容器间启动顺序

---

## 🧑‍💻 作者

Made with ❤️ by Yolk

---
