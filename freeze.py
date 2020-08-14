def get_layer_names_old():
    layer_names = []
    layer_names.append('bw_conv1_1')
    layer_names.append('conv1_2')
    for i in range(2, 9):
        layer_names.append("conv{}_1".format(i))
        layer_names.append("conv{}_2".format(i))
        if i != 2:
            layer_names.append("conv{}_3".format(i))
    return layer_names


def get_layer_names_new():
    layer_names = get_layer_names_old()
    layer_names.append('ab_conv1_1')
    layer_names.append('conv3_3_short')
    layer_names.append('conv2_2_short')
    layer_names.append('conv1_2_short')
    layer_names.append("conv9_1")
    layer_names.append("conv9_2")
    layer_names.append("conv10_1")
    layer_names.append("conv10_2")
    layer_names.append('conv10_ab')
    return layer_names


class Freezer:
    def __init__(self, model):
        self.model = model

    def freeze_layers(self, layers_to_freeze):
        for layer_name in layers_to_freeze:
            self.model.get_layer(layer_name).trainable = False

    def freeze_layers_old(self):
        self.freeze_layers(get_layer_names_old())

    def freeze_layers_new(self):
        self.freeze_layers(get_layer_names_new())



