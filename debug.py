import gin
from rave import RAVE

gin.parse_config_file("original.gin")
model = RAVE()
print(model)