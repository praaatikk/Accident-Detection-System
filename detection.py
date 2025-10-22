# detection.py
from keras.models import model_from_json
import numpy as np
import json
import os

class AccidentDetectionModel(object):
    class_nums = ['Accident', 'No Accident']

    def __init__(self, model_json_file, model_weights_file):
        clean_path = os.path.splitext(model_json_file)[0] + "_clean.json"
        use_path = model_json_file

        # Prefer previously cleaned JSON if available
        if os.path.exists(clean_path):
            use_path = clean_path

        # Load JSON (as Python object)
        with open(use_path, "r") as f:
            try:
                model_obj = json.load(f)
            except json.JSONDecodeError:
                f.seek(0)
                model_obj = json.loads(f.read())

        # Sanitizer: removes legacy keys and turns dtype objects into simple strings
        def sanitize(obj):
            # If dict describing a Keras dtype policy like {"module": "...", "class_name": "DTypePolicy", "config": {"name": "float32"}, ...}
            if isinstance(obj, dict):
                # If it's the wrapped form with 'module' and 'class_name'
                if 'class_name' in obj and obj.get('class_name') == 'DTypePolicy' and isinstance(obj.get('config'), dict):
                    return obj['config'].get('name', 'float32')

                # If nested dict contains a dtype object under keys like 'dtype'
                # Build a new dict skipping legacy keys
                new = {}
                for k, v in obj.items():
                    # remove keys that commonly break new Keras
                    if k in ('batch_shape', 'ragged', 'registered_name'):
                        continue
                    # If we find a dtype field that's itself a dict with 'module' and 'class_name', sanitize it
                    if k == 'dtype' and isinstance(v, dict):
                        # If it's a DTypePolicy structure
                        if v.get('class_name') == 'DTypePolicy' and isinstance(v.get('config'), dict):
                            new[k] = v['config'].get('name', 'float32')
                            continue
                        # otherwise sanitize recursively
                        new[k] = sanitize(v)
                        continue

                    new[k] = sanitize(v)
                return new

            elif isinstance(obj, list):
                return [sanitize(i) for i in obj]
            else:
                return obj

        # If we didn't already use a cleaned file, sanitize and save cleaned copy
        if use_path == model_json_file:
            sanitized = sanitize(model_obj)
            try:
                with open(clean_path, "w") as cf:
                    json.dump(sanitized, cf)
                print(f"[INFO] Cleaned model JSON saved to {clean_path}")
                model_obj = sanitized
                use_path = clean_path
            except Exception as e:
                # fallback: continue with sanitized object in-memory
                model_obj = sanitized
                print(f"[WARN] Could not write cleaned JSON to disk: {e}. Will attempt to load from memory.")

        # Serialize sanitized object to JSON string and load model
        model_json_str = json.dumps(model_obj)
        from keras import mixed_precision

        # Ensure float32 policy is active for all layers
        mixed_precision.set_global_policy("float32")

        self.loaded_model = model_from_json(model_json_str)

        # Load weights
        self.loaded_model.load_weights(model_weights_file)
        self.loaded_model.make_predict_function()
        print("[INFO] Model loaded successfully from:", use_path)

    def predict_accident(self, img):
        preds = self.loaded_model.predict(img)
        return AccidentDetectionModel.class_nums[np.argmax(preds)], preds
