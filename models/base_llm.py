# models/base_llm.py
from abc import ABC, abstractmethod

class BaseLLM(ABC):
    """
    抽象基类 (ABC)，定义了所有大型语言模型 (LLMs) 的通用接口。
    所有具体的 LLM 实现都必须继承此类别并实现其抽象方法。
    """
    def __init__(self, model_name: str, model_args: dict):
        """
        使用其名称和参数初始化基类 LLM。

        参数:
            model_name (str): 模型的具体名称（例如，'Claude-3-Opus'、'Llama-3-70B'）。
            model_args (dict): 特定于此模型的参数字典，
                                例如 API 密钥、模型路径等。
        """
        self.model_name = model_name
        self.model_args = model_args

    @abstractmethod
    def get_response(self, messages: list) -> str:
        """
        从 LLM 获取响应的抽象方法。
        具体实现必须提供与其特定 LLM API 或本地模型交互的逻辑。

        参数:
            messages (list): 一个消息字典列表，通常以聊天式格式呈现。

        返回:
            str: 生成的文本响应。
        """
        pass

    @abstractmethod
    def get_token_count(self, messages: list) -> int:
        """
        计算给定消息集 token 数量的抽象方法。
        具体实现必须提供针对其 LLM 的消息分词逻辑。

        参数:
            messages (list): 要计算 token 的消息字典列表。

        返回:
            int: token 的数量。
        """
        pass