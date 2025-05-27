# prompt.py

class PromptConfig:
    """
    A class for storing and managing various AI prompts.
    Prompts are stored as a dictionary for easy access via key-value pairs.
    """
    _prompts = {
        "example": "example",
        # Add more prompts as needed
    }

    @classmethod
    def get_prompt(cls, prompt_name: str, **kwargs) -> str:
        """
        Retrieves the prompt string by its name.
        Allows for dynamic content injection into the prompt using keyword arguments.

        Args:
            prompt_name (str): The name of the prompt to retrieve.
            **kwargs: Keyword arguments for customizing the prompt's content.

        Returns:
            str: The corresponding prompt string, or an empty string if not found.
        """
        prompt_template = cls._prompts.get(prompt_name, "")
        if not prompt_template:
            print(f"Warning: Prompt '{prompt_name}' not found.")
            return ""

        try:
            return prompt_template.format(**kwargs)
        except KeyError as e:
            print(f"Missing required content for prompt: {e}")
            return ""

    @classmethod
    def add_prompt(cls, prompt_name: str, prompt_content: str):
        """
        Adds or updates a prompt.

        Args:
            prompt_name (str): The name of the prompt to add or update.
            prompt_content (str): The content of the prompt.
        """
        cls._prompts[prompt_name] = prompt_content
        print(f"Prompt '{prompt_name}' has been added/updated.")

    @classmethod
    def list_prompts(cls):
        """
        Lists the names of all stored prompts.
        """
        print("Currently stored prompt names:")
        for name in cls._prompts.keys():
            print(f"- {name}")