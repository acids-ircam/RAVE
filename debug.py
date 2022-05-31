from rave.blocks import Generator, Encoder
import gin

gin.parse_config_file("default.gin")

g = Generator(16, 64, 16, [4, 4, 4, 2], 1, True)
print(g)

e = Encoder(16, 64, 16, [4, 4, 4, 2])
print(e)