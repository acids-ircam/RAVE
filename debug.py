import torch
import gin
from rave.core import EncodecAudioDistance


@gin.configurable
def get_distance(dist: EncodecAudioDistance) -> EncodecAudioDistance:
    return dist


gin.parse_config_file('rave/configs/discrete.gin')
gin.parse_config('get_distance.dist = @core.EncodecAudioDistance()')

dist = get_distance()

x = torch.randn(1, 1, 2**16)
y = torch.randn(1, 1, 2**16)

print(dist(x, y))