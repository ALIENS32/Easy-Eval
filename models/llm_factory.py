# models/llm_factory.py
import importlib
# 注意：这里不再需要 from models import *，因为 BaseLLM 会从相对路径导入

class LLM:
    """
    大型语言模型 (LLM) 的工厂类。
    它根据给定的模型名称动态加载相应的模型类。
    """
    def __init__(self, model_name: str, model_args: dict):
        """
        通过动态加载指定模型来初始化 LLM 实例。

        参数:
            model_name (str): 要加载的 LLM 名称（例如，'Claude'、'Llama'、'Gemini'）。
                              此名称应对应于 'models/implementations/' 目录中的模块和类名。
            model_args (dict): 针对所加载模型的特定参数字典。
                                这些参数将直接传递给模型的构造函数。
        """
        self.model_name = model_name
        # 动态获取模型实例
        self.model = self._get_model(model_name, model_args)

    def _get_model(self, model_name: str, model_args: dict):
        """
        私有方法，用于动态导入并实例化特定的 LLM 类。

        参数:
            model_name (str): 要导入的模型名称。
            model_args (dict): 模型的构造函数参数。

        返回:
            指定 LLM 类的实例，该实例必须继承自 BaseLLM。

        抛出:
            ImportError: 如果找不到给定 model_name 的模块或类。
            AttributeError: 如果在导入的模块中找不到 model_name 类。
        """
        try:
            # 关键改动：从 'models.implementations' 包中导入模块
            # 示例：对于 model_name='Claude'，它会尝试导入 'models.implementations.Claude'。
            module = importlib.import_module(f'models.implementations.{model_name}')
            # 从导入的模块中获取类定义。
            # 期望类名与 model_name 相同。
            model_class = getattr(module, model_name)
            # 实例化模型类与它的特定参数。
            return model_class(model_name, model_args)
        except (ImportError, AttributeError) as e:
            raise RuntimeError(f"加载模型 '{model_name}' 失败: {e}。 "
                               f"请确保 'models/implementations/{model_name}.py' 存在并包含名为 '{model_name}' 的类。")

    def get_response(self, messages: list) -> str:
        """
        从已加载的 LLM 中获取响应。

        参数:
            messages (list): 一个消息字典列表，格式需由特定 LLM 理解。
                             通常遵循 OpenAI 聊天补全格式（例如，[{"role": "user", "content": "..."}]）。

        返回:
            str: LLM 生成的响应。
        """
        return self.model.get_response(messages)

    def get_token_count(self, messages: list) -> int:
        """
        使用已加载 LLM 的分词器计算给定消息集的 token 数量。

        参数:
            messages (list): 要计算 token 的消息字典列表。

        返回:
            int: 消息中的 token 数量。
        """
        return self.model.get_token_count(messages)