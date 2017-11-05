from networking import FeatureComputer, default_config, load_config


def run_vgg16():
    from keras.applications import VGG16
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.3
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))

    settings = load_config('server.yaml')
    model = VGG16(False, "imagenet")
    print("Running vgg16")

    z = FeatureComputer(bind_str=settings['bind_addr'], parent_model=model, logins=settings['logins'])
    z.run()

def run_resnet50():
    from keras.applications import ResNet50
    import keras
    settings = load_config('server.yaml')
    model = ResNet50(False, "imagenet")

    # ResNet has a crapload of layers, and most of them aren't really that interesting to look at
    viewable_layers = [layer for layer in model.layers if isinstance(layer, keras.layers.core.Activation)]

    z = FeatureComputer(bind_str=settings['bind_addr'], parent_model=model, logins=settings['logins'], viewable_layers=viewable_layers)
    z.run()

if __name__ == "__main__":
    run_vgg16()
   # run_resnet50()
