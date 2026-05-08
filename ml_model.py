import os
import numpy as np
import logging

log = logging.getLogger(__name__)

TFLITE_MODEL_PATH = "tremor_model.tflite"


class TremorMLModel:

    def __init__(self):

        self.interpreter = None

        self.input_details = None

        self.output_details = None

        self.load_model()


    def load_model(self):

        if not os.path.exists(TFLITE_MODEL_PATH):

            log.warning("TFLite model not found → using fallback rule system")

            return


        try:

            import tensorflow as tf

            self.interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)

        except:

            log.warning("Tensorflow not found → fallback mode")

            return


        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()

        self.output_details = self.interpreter.get_output_details()

        log.info("TFLite model loaded")


    def create_features(self, tremor_pct, hr_bpm, combined_pct):

        tremor = tremor_pct / 100

        hr = hr_bpm / 200

        combined = combined_pct / 100

        hr_dev = abs(hr_bpm - 72) / 50

        interaction = tremor * hr_dev

        return np.array([[tremor, hr, combined, hr_dev, interaction]], dtype=np.float32)


    def predict(self, tremor_pct, hr_bpm, combined_pct):

        if self.interpreter:

            features = self.create_features(tremor_pct, hr_bpm, combined_pct)

            self.interpreter.set_tensor(self.input_details[0]['index'], features)

            self.interpreter.invoke()

            output = self.interpreter.get_tensor(self.output_details[0]['index'])[0]


            weights = [0, 30, 65, 100]

            severity = float(np.dot(output, weights))


        else:

            # fallback simple formula

            severity = (tremor_pct * 0.7) + (abs(hr_bpm - 72) * 0.3)


        if severity < 15:

            label = "Normal"

        elif severity < 40:

            label = "Mild"

        elif severity < 70:

            label = "Moderate"

        else:

            label = "Severe"


        return {

            "severity": round(severity, 2),

            "label": label

        }