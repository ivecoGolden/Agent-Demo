import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import uuid
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_register_and_login():
    email = f"{uuid.uuid4().hex}@example.com"
    # 注册
    res = client.post(
        "/api/register",
        json={
            "username": email,
            "email": email,
            "password": "test1234",
        },
    )
    assert res.status_code == 200
    assert res.json()["code"] == 0
    user_id = res.json()["data"]["id"]

    # 登录
    res = client.post(
        "/api/login",
        json={"username": email, "password": "test1234"},
    )
    print(res.json())
    assert res.status_code == 200
    assert res.json()["code"] == 0
    assert "access_token" in res.json()["data"]

    token = res.json()["data"]["access_token"]

    # 获取当前用户信息
    res = client.get("/api/me", headers={"Authorization": f"Bearer {token}"})
    assert res.status_code == 200
    assert res.json()["code"] == 0
    assert res.json()["data"]["id"] == user_id
