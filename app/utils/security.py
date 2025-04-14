from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    """使用bcrypt算法对密码进行哈希处理

    Args:
        password (str): 需要哈希的原始密码字符串

    Returns:
        str: 经过bcrypt哈希处理后的密码字符串

    Note:
        - 使用passlib库的CryptContext实现
        - 自动生成salt并包含在哈希结果中
        - 默认使用bcrypt算法(12轮加密)
    """
    return pwd_context.hash(password)  # 生成密码哈希值


def verify_password(raw_password: str, hashed: str) -> bool:
    """验证原始密码与哈希密码是否匹配

    Args:
        raw_password (str): 待验证的原始密码
        hashed (str): 存储的哈希密码

    Returns:
        bool: 验证结果，True表示匹配，False表示不匹配

    Note:
        - 使用bcrypt算法进行安全验证
        - 自动处理salt值比较
        - 防止时序攻击(timeing attack)
    """
    return pwd_context.verify(raw_password, hashed)  # 安全验证密码
