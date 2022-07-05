# %%
import rave
import gin

gin.parse_config_file("configs/qrave.gin")
model = rave.RAVE()

embeds = map(lambda l: l._codebook.embed, model.encoder.rvq.layers)

for e in embeds:
    print(e.shape)
# %%
