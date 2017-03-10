from networking import FeatureComputer, default_config, load_config
from keras.applications import VGG16

def run_vgg16():
    settings = load_config('server.yaml')
    model = VGG16(False, "imagenet")
    z = FeatureComputer(bind_str=settings['bind_addr'], parent_model=model, logins=settings['logins'])
    z.run()

if __name__ == "__main__":
    run_vgg16()
