from openai import OpenAI


class gpt_4_1:
    def __init__(self, model_name, model_args):
        """
        Initializes the GPT-4.1 model with the given name and arguments.
        
        Args:
            model_name (str): The name of the model, e.g., 'gpt-4-1'.
            model_args (dict): A dictionary of arguments specific to the model.
        """
        self.model_name = model_name
        self.model_args = model_args
        self.client = OpenAI(
            base_url=model_args.get('base_url'),
            api_key=model_args.get('api_key')
        )
        self.completion_kwargs={
            # 添加要传入的参数
        }
    
    def get_response(self, messages):
        """
        Orchestrates the chat completion process and returns the formatted response.
        """
        self.completion_kwargs["messages"] = messages
        completion = self.client.chat.completions.create(**self.completion_kwargs)

        if self.model_args.stream:
            response_data = self._handle_streamed_completion(completion)
        else:
            response_data = self._handle_non_streamed_completion(completion)

        self.total_token_cnt += response_data["all_token_count"]
        return response_data

    def _handle_streamed_completion(self, completion):
        """
        Processes a streamed chat completion response.
        """
        return_template = {
            "formal_answer": "",
            "all_token_count": 0,
            "thinking_content": "",
            "thinking_token_count": 0,
        }
        last_chunk = None
        for chunk in completion:
            last_chunk = chunk
            if chunk.choices and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                if hasattr(delta, "reasoning_content") and delta.reasoning_content is not None:
                    return_template["thinking_content"] += delta.reasoning_content
                if hasattr(delta, "content") and delta.content:
                    return_template["formal_answer"] += delta.content

        if last_chunk:
            return_template["all_token_count"] = last_chunk.usage.completion_tokens
            if hasattr(last_chunk.usage, 'completion_tokens_details') and \
               hasattr(last_chunk.usage.completion_tokens_details, 'reasoning_tokens'):
                return_template["thinking_token_count"] = last_chunk.usage.completion_tokens_details.reasoning_tokens
        return return_template

    def _handle_non_streamed_completion(self, completion):
        """
        Processes a non-streamed chat completion response.
        """
        return_template = {
            "formal_answer": "",
            "all_token_count": 0,
            "thinking_content": "", # For non-streaming, thinking_content might not be directly available or streamed.
            "thinking_token_count": 0,
        }
        return_template["all_token_count"] = completion.usage.completion_tokens
        return_template["formal_answer"] = completion.choices[0].message.content.strip()

        try:
            # Prioritize reasoning from completion.usage.completion_tokens_details if available
            if hasattr(completion.usage, 'completion_tokens_details') and \
               hasattr(completion.usage.completion_tokens_details, 'reasoning_tokens'):
                return_template["thinking_token_count"] = completion.usage.completion_tokens_details.reasoning_tokens
            # Fallback to message.reasoning if details not present and reasoning is directly on message
            elif hasattr(completion.choices[0].message, 'reasoning') and \
                 isinstance(completion.choices[0].message.reasoning, int): # Ensure it's a token count
                return_template["thinking_token_count"] = completion.choices[0].message.reasoning
        except Exception:
            # Handle potential errors during attribute access gracefully
            pass
        return return_template
        