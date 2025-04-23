from app.core.config import settings
from jose import JWTError, jwt
from datetime import datetime, timedelta

SECRET_KEY = settings.SECRET_KEY
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = settings.ACCESS_TOKEN_EXPIRE_MINUTES


def create_access_token(data: dict, expires_delta: timedelta | None = None):
    """创建JWT访问令牌

    Args:
        data (dict): 需要编码到令牌中的声明数据
        expires_delta (timedelta | None): 可选，令牌有效期时间差

    Returns:
        str: 编码后的JWT令牌字符串

    Note:
        - 会自动添加exp过期时间声明
        - 使用HS256算法和配置的密钥进行签名
    """
    to_encode = data.copy()  # 复制输入数据避免修改原始字典
    expire = datetime.utcnow() + (
        expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    to_encode.update({"exp": expire})  # 添加exp过期时间声明
    return jwt.encode(
        to_encode, SECRET_KEY, algorithm=ALGORITHM
    )  # 使用HS256算法编码令牌


def decode_access_token(token: str):
    """解码并验证JWT访问令牌

    Args:
        token (str): 需要解码的JWT令牌字符串

    Returns:
        dict | None: 解码后的声明数据字典，如果令牌无效则返回None

    Note:
        - 使用相同的密钥和算法(HS256)进行验证
        - 会自动验证令牌签名和过期时间
        - 捕获所有JWT相关异常并返回None
    """
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])  # 解码并验证令牌
    except JWTError:  # 捕获所有JWT相关异常(签名无效、过期等)
        return None  # 令牌无效时返回None
