import os
import re
import json
from typing import Any, Optional

class Utils:
    """
    A utility class containing various helper methods for common tasks
    like data extraction, formatting, or other general operations.
    """

    @staticmethod
    def extract_json_data(response_text: str) -> Optional[Any]:
        """
        Extracts JSON data from a text response that is enclosed within
        a '```json ... ```' code block.

        This method searches for a JSON string wrapped by '```json' and '```'
        (e.g., from a large language model's output) and attempts to parse it
        into a Python object (dictionary or list).

        Args:
            response_text (str): The complete text response to search within.

        Returns:
            Optional[Any]: The parsed Python object (dict or list) if valid JSON is found
                           and successfully parsed; otherwise, returns None.
        """
        # Use a regular expression to find content between '```json' and '```'.
        # re.DOTALL allows '.' to match newlines as well.
        match = re.search(r'```json\s*([\s\S]*?)\s*```', response_text, re.DOTALL)

        if match:
            json_string = match.group(1).strip() # Extract the matched JSON string and strip whitespace
            try:
                # Attempt to parse the JSON string
                return json.loads(json_string)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON from response: {e}")
                print(f"Problematic JSON snippet: \n{json_string[:200]}...") # Print a part of the problematic string
                return None
        else:
            # No '```json ... ```' block found
            return None

    @staticmethod
    def read_from_jsonl(file_path):
        """从JSONL文件中读取数据"""
        return [json.loads(line.strip()) for line in open(file_path, 'r', encoding='utf-8') if line.strip()] if os.path.exists(file_path) else []
    
    @staticmethod
    def write_to_jsonl(file_path, data):
        """将数据写入JSONL文件"""
        with open(file_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')