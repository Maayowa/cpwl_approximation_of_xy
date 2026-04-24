import sys
import enum
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import yaml
from .tempregpy.logging_config import *

# from scipy.spatial import ConvexHull
import pandas as pd
import traceback


def load_cascade_network(net_path="", inputs=None):
    try:

        inputs = {} if inputs is None else inputs
        with open(net_path, "r") as f:
            G = yaml.load(f, Loader=yaml.Loader)
        logger.info("Network loaded succesfully")
        inputs["network"] = G
        return inputs
    except Exception as e:
        logger.error(f"Not able to parse Inputs. Error: {e}")
        logger.error("Stack trace:\n" + traceback.format_exc())
        sys.exit(1)
        logger.error(f"Not able to parse Inputs. Error: {e}")
        sys.exit(1)


def load_hydro_inputs_from_excel(config_path):
    inputs = {}
    excel_file = pd.ExcelFile(config_path)

    # Expected sheets -> Pricing, Storage to Elevation, Power Curves, Config

    # excel_data = excel_file.parse("Inputs")

    inputs["pricing"] = excel_file.parse("Pricing")
    inputs["elevation"] = excel_file.parse("Storage to Elevation")
    inputs["config"] = excel_file.parse("Config").set_index("Plant")
    inputs["curves"] = excel_file.parse("Power Curves")

    return inputs
