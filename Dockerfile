FROM python:3.13-slim
RUN apt-get update && apt-get install -y curl
# 设置工作目录
WORKDIR /app

# 安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制代码
COPY . .

# 添加执行权限给 wait-for-it.sh
RUN chmod +x wait-for-it.sh

# 启动应用
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "app.main:app", "--bind", "0.0.0.0:7000"]