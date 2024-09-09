import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("GPU detected. Installing TensorFlow with CUDA support.")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tensorflow[cuda]"])
