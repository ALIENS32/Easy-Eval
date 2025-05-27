import argparse
from config.model_args_config import ModelArgsConfig
import os
import sys
from dataclasses import asdict
# 添加当前目录的上级目录到系统路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, '..')))
from models.llm_factory import LLM
from utils import Utils













if __name__ == "__main__":
    # args
    parser = argparse.ArgumentParser(
        description=""
    )
    parser.add_argument("--model_name", type=str, default="Qwen3-235B-A22B_thinking")
    parser.add_argument("--evaluator", type=str, default="o3-mini-high")
    parser.add_argument("--save_frequency", type=int, default=1)
    parser.add_argument("--infer_proc", type=int, default=20)
    parser.add_argument("--infer_num", type=bool, default=False, help="")
    # 将 ModelArgsConfig 的参数添加到同一个解析器中
    parser = ModelArgsConfig.get_arg_parser(parser)
    # 解析所有命令行参数
    args = parser.parse_args()

    # 从解析的参数中创建 ModelArgsConfig 实例
    model_args = asdict(ModelArgsConfig.from_argparse_args(args))

    # 实例化模型类
    llm=LLM(model_args) 

    # 创建存储结构的目录
    infer_path = os.path.join("results", args.model_name, f"infer_result.jsonl")
    save_path = os.path.join("results", args.model_name, "eval_result.jsonl")
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    Utils.write_to_jsonl(infer_path, []) # 写入空列表以初始化文件
    # 读取数据
    infer_results = Utils.read_from_jsonl(infer_path)
    existing_eval_results = Utils.read_from_jsonl(save_path)

    # 只评估没有评估过的（通过pid判断）
    evaled_pid=set([data["pid"] for data in existing_eval_results])
    for res in infer_results:
        if res["pid"] not in evaled_pid:
            uneval







