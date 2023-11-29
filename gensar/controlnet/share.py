"""@doem97: so stupid design here. Why the hell author use a seperate file for such trivial thing??"""
import config
from cldm.hack import disable_verbosity, enable_sliced_attention


def setup_config():
    disable_verbosity()

    if config.save_memory:
        enable_sliced_attention()
