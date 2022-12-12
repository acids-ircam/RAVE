import rave
import gin

gin.parse_config_file('/data/antoine/horv2/runs/horv2_1221_dd5d50644e/config.gin')

model = rave.RAVE()

print(model)