from rave.blocks import Generator, Encoder
from rave.discriminator import FullDiscriminator
import gin

gin.parse_config_file("default.gin")

d = FullDiscriminator()

print(d)