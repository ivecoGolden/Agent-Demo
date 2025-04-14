def success(data=None, message="操作成功", code=0):
    """构建成功响应格式

    Args:
        data (Any, optional): 返回的业务数据，默认为None
        message (str, optional): 成功提示信息，默认为"操作成功"
        code (int, optional): 状态码，默认为0表示成功

    Returns:
        dict: 标准化的成功响应格式，包含:
            - code: 状态码
            - message: 提示信息
            - data: 业务数据

    Example:
        >>> success(data={"id": 1}, message="创建成功")
        {'code': 0, 'message': '创建成功', 'data': {'id': 1}}
    """
    return {
        "code": code,  # 状态码
        "message": message,  # 提示信息
        "data": data,  # 返回的业务数据
    }


def error(message="操作失败", code=1):
    """构建错误响应格式

    Args:
        message (str, optional): 错误提示信息，默认为"操作失败"
        code (int, optional): 错误状态码，默认为1表示失败

    Returns:
        dict: 标准化的错误响应格式，包含:
            - code: 错误状态码
            - message: 错误信息
            - data: 固定为None

    Example:
        >>> error(message="用户不存在", code=404)
        {'code': 404, 'message': '用户不存在', 'data': None}
    """
    return {
        "code": code,  # 错误状态码
        "message": message,  # 错误提示信息
        "data": None,  # 错误响应不返回业务数据
    }
