from rave.blocks import Generator, Encoder
from rave.discriminator import FullDiscriminator
import gin

gin.parse_config_file("default.gin")

g = Generator(128, 64, 16, [4, 4, 4, 2], 1, True)
e = Encoder(16, 64, 128, [4, 4, 4, 2])
d = FullDiscriminator()
