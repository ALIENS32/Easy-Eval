# data_template.py

class DataTemplate:
    @classmethod
    def get_infer_result_template(pid) -> dict:
        return {
            "pid":pid
        }
    
    @classmethod
    def get_eval_result_template(pid) -> dict:
        return {
            "pid":pid
        }
