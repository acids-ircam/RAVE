from rave import RAVE
import gin

gin.parse_config_file('configs/rave_onnx.gin')

model = RAVE()