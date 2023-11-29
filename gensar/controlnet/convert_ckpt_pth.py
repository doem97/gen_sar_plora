from share import setup_config
import config
import torch
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler


setup_config()

model = create_model("./models/cldm_v15.yaml").cpu()
# model.load_state_dict(load_state_dict('./models/control_sd15_canny.pth', location='cuda'))
model.load_state_dict(
    load_state_dict("/workspace/dso/gensar/controlnet/checkpoints/fusrs_v2_lc/fusrs_epoch=88.ckpt", location="cpu")
)

torch.save(model.state_dict(), '/workspace/dso/gensar/controlnet/checkpoints/fusrs_v2_lc/fusrs_epoch=88.pth')