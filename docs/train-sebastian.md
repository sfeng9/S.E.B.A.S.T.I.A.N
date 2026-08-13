# Train the Sebastian wake word

Sebastian is not bundled with openWakeWord, so it needs a custom ONNX model.

1. Open the [official simple openWakeWord Colab](https://colab.research.google.com/drive/1q1oe2zOyZp7UsB3jJiQ1IFn8z5YfjwEb?usp=sharing).
2. Sign in to Google and save a copy of the notebook to Drive when prompted.
3. In Colab, select **Runtime > Change runtime type**, then choose a T4 GPU.
4. Run the setup cells in order.
   If the notebook reports `ModuleNotFoundError: No module named 'webrtcvad'`,
   insert and run this cell immediately before the failing cell:

   ```python
   %pip install -q webrtcvad-wheels
   ```

   Then rerun the cell that imports `generate_samples`. The wheel package keeps
   the expected `import webrtcvad` module name and supports current Colab Python.
5. Enter `sebastian` as the model name and `Sebastian` as the target phrase. In the detailed notebook configuration, the equivalent values are:

   ```python
   config["model_name"] = "sebastian"
   config["target_phrase"] = ["Sebastian"]
   ```

6. Run the remaining cells. Basic training normally takes less than an hour, but Colab availability and notebook dependencies can change.
7. Download the generated `sebastian.onnx` file. The `.tflite` file is not used by this Windows assistant.
8. Put the ONNX file at `data/wake_words/sebastian.onnx` in this project.
9. Prepare the runtime and test the model:

   ```powershell
   .\.venv\Scripts\python.exe .\tools\setup_wake_word.py
   .\.venv\Scripts\python.exe .\tools\test_wake_word.py --timeout 30
   ```

Say "Sebastian" several times from a normal distance. If detection is unreliable,
adjust `wake_word.threshold` in `config/assistant.json` in small increments. Lower
values detect more easily but increase false activations; higher values do the
opposite.
