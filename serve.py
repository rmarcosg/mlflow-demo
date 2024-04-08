import mlflow
from mlflow.models import ModelSignature
import json


model_input = json.dumps([{'name': 'text', 'type': 'string'}])
model_output = json.dumps([{'name': 'text', 'type': 'string'}])
signature = ModelSignature.from_dict({'inputs': model_input, 'outputs': model_output})

mlflow.set_tracking_uri("http://127.0.0.1:5000")


class Summarizer(mlflow.pyfunc.PythonModel):

    def __init__(self):
        from transformers import T5Tokenizer, T5ForConditionalGeneration

        self.tokenizer = T5Tokenizer.from_pretrained("t5-small")
        self.model = T5ForConditionalGeneration.from_pretrained("t5-small")

    def predict(self, context, model_input):
        input_ids = self.tokenizer(f"translate English to German: {model_input}", return_tensors="pt").input_ids
        outputs = self.model.generate(input_ids)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


# Start tracking
with mlflow.start_run(run_name="t5-summarizer") as run:
    print(run.info.run_id)
    runner = run.info.run_id
    mlflow.pyfunc.log_model('model', loader_module=None, data_path=None, code_path=None,
                            conda_env=None, python_model=Summarizer(),
                            artifacts=None, registered_model_name="t5-small-summarizer", signature=signature,
                            input_example=None, await_registration_for=0)
