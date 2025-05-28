import argparse
import json
from dataclasses import dataclass, field
from typing import Any, Optional, List, Dict, Type, TypeVar

T = TypeVar('T', bound='ModelArgsConfig')

@dataclass
class ModelArgsConfig:
    """
    Configuration class for large language model arguments.
    This class uses dataclasses for type hinting and default values,
    making it easy to define and manage model parameters.
    """
    api_key: Optional[str] = field(
        default=None,
    )
    base_url: Optional[str] = field(
        default=None,
    )
    temperature: float = field(
        default=1e-6,
        metadata={"help": "Controls the randomness of the output. Higher values mean more random."}
    )
    max_tokens: Optional[int] = field(
        default=None,
        metadata={"help": "The maximum number of tokens to generate in the completion."}
    )
    stream: bool = field(
        default=False,
        metadata={"help": "Whether to stream back partial progress. If set, tokens will be sent as data-only server-sent events."}
    )
    # ... (其他 ModelArgsConfig 参数保持不变) ...
    stop: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "Up to 4 sequences where the API will stop generating further tokens. "
                    "Provide as a comma-separated string, e.g., 'stop1,stop2'."
        },
    )
    presence_penalty: float = field(
        default=0.0,
        metadata={
            "help": "Positive values penalize new tokens based on whether they appear in the text so far, "
                    "increasing the model's likelihood to talk about new topics."
        },
    )
    frequency_penalty: float = field(
        default=0.0,
        metadata={
            "help": "Positive values penalize new tokens based on their existing frequency in the text so far, "
                    "decreasing the model's likelihood to repeat the same line verbatim."
        },
    )
    logit_bias: Optional[Dict[int, float]] = field(
        default=None,
        metadata={
            "help": "Modify the likelihood of specified tokens appearing in the completion. "
                    "Maps token IDs to an associated bias value from -100 to 100. "
                    "Provide as a JSON string, e.g., '{\"123\": 10, \"456\": -5}'."
        },
    )
    user: Optional[str] = field(
        default=None,
        metadata={
            "help": "A unique identifier representing your end-user, which can help OpenAI to monitor and detect abuse."
        },
    )
    seed: Optional[int] = field(
        default=None,
        metadata={
            "help": "If specified, our system will make a best effort to sample deterministically, "
                    "such that repeated requests with the same seed and parameters should return the same result."
        },
    )
    response_format: Optional[Dict[str, str]] = field(
        default=None,
        metadata={
            "help": "An object specifying the format that the model must output. "
                    "E.g., '{\"type\": \"json_object\"}' for JSON mode."
        },
    )
    tools: Optional[List[Dict[str, Any]]] = field(
        default=None,
        metadata={
            "help": "A list of tools the model may call. Currently only function calling is supported. "
                    "Provide as a JSON string representing a list of tool definitions."
        },
    )
    tool_choice: Optional[str] = field(
        default=None,
        metadata={
            "help": "Controls which (if any) tool is called by the model. 'none', 'auto', or a specific tool name string."
        },
    )
    top_p: float = field(
        default=1.0,
        metadata={
            "help": "Controls nucleus sampling. Only considers tokens whose cumulative probability exceeds top_p."
        },
    )
    n: int = field(
        default=1,
        metadata={
            "help": "How many completions to generate for each prompt."
        },
    )


    @classmethod
    def from_argparse_args(cls: Type[T], args: argparse.Namespace) -> T:
        args_dict = vars(args)
        init_kwargs = {}
        for f in field(cls):
            if f.name in args_dict and args_dict[f.name] is not None:
                arg_value = args_dict[f.name]
                if f.name == 'stop' and isinstance(arg_value, str):
                    init_kwargs[f.name] = [s.strip() for s in arg_value.split(',')] if arg_value else None
                elif f.name in ['logit_bias', 'response_format', 'tools'] and isinstance(arg_value, str):
                    try:
                        init_kwargs[f.name] = json.loads(arg_value)
                    except json.JSONDecodeError:
                        print(f"Warning: Could not parse JSON for {f.name}: '{arg_value}'. Skipping.")
                        continue
                else:
                    init_kwargs[f.name] = arg_value
        return cls(**init_kwargs)

    @classmethod
    def get_arg_parser(cls, parser: Optional[argparse.ArgumentParser] = None) -> argparse.ArgumentParser:
        if parser is None:
            parser = argparse.ArgumentParser(description="Configure LLM parameters via command line.")

        for field_name, field_obj in cls.__dataclass_fields__.items():
            field_type = field_obj.type
            default_value = field_obj.default
            help_text = field_obj.metadata.get("help", "")

            actual_type = field_type
            if hasattr(field_type, '__origin__') and field_type.__origin__ is Optional:
                actual_type = field_type.__args__[0]

            if actual_type is bool:
                if default_value is True:
                    parser.add_argument(
                        f'--no-{field_name.replace("_", "-")}',
                        action='store_false',
                        dest=field_name,
                        help=f"Disable {help_text.lower().replace('whether to ', '')}"
                    )
                else:
                    parser.add_argument(
                        f'--{field_name.replace("_", "-")}',
                        action='store_true',
                        help=help_text
                    )
            else:
                arg_type = str if actual_type in (list, dict) else actual_type
                argparse_default = None
                if default_value is not field_obj.default_factory and default_value is not field_obj.default:
                    argparse_default = default_value

                parser.add_argument(
                    f'--{field_name.replace("_", "-")}',
                    type=arg_type,
                    default=argparse_default,
                    help=help_text
                )
        return parser