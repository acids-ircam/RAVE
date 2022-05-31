import gin
from rave import RAVE

gin.parse_config_file("default.gin")

model = RAVE()

print(model)