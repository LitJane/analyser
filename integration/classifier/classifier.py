from mlflow.pyfunc import PythonModel, PythonModelContext
import pandas as pd

from integration.classifier.text_utils import cleanup_all


class PracticeClassifier(PythonModel):
    def load_context(self, context: PythonModelContext):
        import os
        from transformers import AutoTokenizer, AutoConfig
        from transformers import TFAutoModelForSequenceClassification
        # noinspection PyUnresolvedReferences
        import tensorflow as tf

        config_file = os.path.dirname(context.artifacts["config"])
        self.config = AutoConfig.from_pretrained(config_file)
        self.tokenizer = AutoTokenizer.from_pretrained(config_file)
        self.model = TFAutoModelForSequenceClassification.from_pretrained(config_file, config=self.config)

        # _ = self.model.eval()

    def predict(self, context: PythonModelContext, data: pd.DataFrame) -> pd.DataFrame:
        import tensorflow as tf
        import pandas as pd

        input_strings = [cleanup_all(x) for x in data['text'].values]
        inputs = self.tokenizer(input_strings, truncation=True, padding=True, max_length=512, return_tensors='tf')

        predictions = self.model(**inputs)['logits']
        probs = tf.nn.softmax(predictions, axis=1).numpy()

        classes = probs.argmax(axis=1)
        confidences = probs.max(axis=1)

        return pd.DataFrame({'practice': [self.config.id2label[c] for c in classes],
                             'confidence': confidences})
