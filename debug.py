import prior
from prior import core
import gin

gin.parse_config_file("configs/prior.gin")
model = prior.Prior()

print(core.get_prior_receptive_field(model))