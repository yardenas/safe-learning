import argparse
import os

import ss2r.benchmark_suites.mujoco_playground.g1_mocap_tracking.loco_mujoco as loco_mujoco
import yaml


def set_amass_path():
    """
    Set the path to the AMASS dataset.
    """
    parser = argparse.ArgumentParser(description="Set the AMASS dataset path.")
    parser.add_argument("--path", type=str, help="Path to the AMASS dataset.")
    args = parser.parse_args()
    _set_path_in_yaml_conf(
        args.path, "LOCOMUJOCO_AMASS_PATH", path_to_conf=loco_mujoco.PATH_TO_VARIABLES
    )


def set_smpl_model_path():
    """
    Set the path to the SMPL model.
    """
    parser = argparse.ArgumentParser(description="Set the SMPL model path.")
    parser.add_argument("--path", type=str, help="Path to the SMPL model.")
    args = parser.parse_args()
    _set_path_in_yaml_conf(
        args.path,
        "LOCOMUJOCO_SMPL_MODEL_PATH",
        path_to_conf=loco_mujoco.PATH_TO_VARIABLES,
    )


def set_all_caches():
    """
    Set the path to which all converted datasets will be stored. This sets the following Variables:
    - LOCOMUJOCO_CONVERTED_AMASS_PATH
    - LOCOMUJOCO_CONVERTED_LAFAN1_PATH
    - LOCOMUJOCO_CONVERTED_DEFAULT_PATH

    Returns:

    """
    parser = argparse.ArgumentParser(
        description="Set the path to which all converted datasets will be stored."
    )
    parser.add_argument(
        "--path", type=str, help="Path to which all converted datasets will be stored."
    )
    args = parser.parse_args()
    amass_path = os.path.join(args.path, "AMASS")
    _set_path_in_yaml_conf(
        amass_path,
        "LOCOMUJOCO_CONVERTED_AMASS_PATH",
        path_to_conf=loco_mujoco.PATH_TO_VARIABLES,
    )
    lafan1_path = os.path.join(args.path, "LAFAN1")
    _set_path_in_yaml_conf(
        lafan1_path,
        "LOCOMUJOCO_CONVERTED_LAFAN1_PATH",
        path_to_conf=loco_mujoco.PATH_TO_VARIABLES,
    )
    default_path = os.path.join(args.path, "DEFAULT")
    _set_path_in_yaml_conf(
        default_path,
        "LOCOMUJOCO_CONVERTED_DEFAULT_PATH",
        path_to_conf=loco_mujoco.PATH_TO_VARIABLES,
    )


def set_converted_amass_path():
    """
    Set the path to which the converted AMASS dataset is stored.
    """
    parser = argparse.ArgumentParser(
        description="Set the path to which the converted AMASS dataset is stored."
    )
    parser.add_argument(
        "--path", type=str, help="Path to which the converted AMASS dataset is stored."
    )
    args = parser.parse_args()
    _set_path_in_yaml_conf(
        args.path,
        "LOCOMUJOCO_CONVERTED_AMASS_PATH",
        path_to_conf=loco_mujoco.PATH_TO_VARIABLES,
    )


def set_lafan1_path():
    """
    Set the path to the LAFAN1 dataset.
    """
    parser = argparse.ArgumentParser(description="Set the LAFAN1 dataset path.")
    parser.add_argument("--path", type=str, help="Path to the LAFAN1 dataset.")
    args = parser.parse_args()
    _set_path_in_yaml_conf(
        args.path, "LOCOMUJOCO_LAFAN1_PATH", path_to_conf=loco_mujoco.PATH_TO_VARIABLES
    )


def set_converted_lafan1_path():
    """
    Set the path to which the converted LAFAN1 dataset is stored.
    """
    parser = argparse.ArgumentParser(
        description="Set the path to which the converted LAFAN1 dataset is stored."
    )
    parser.add_argument(
        "--path", type=str, help="Path to which the converted LAFAN1 dataset is stored."
    )
    args = parser.parse_args()
    _set_path_in_yaml_conf(
        args.path,
        "LOCOMUJOCO_CONVERTED_LAFAN1_PATH",
        path_to_conf=loco_mujoco.PATH_TO_VARIABLES,
    )


def _set_path_in_yaml_conf(path: str, attr: str, path_to_conf: str):
    """
    Set the path in the yaml configuration file.
    """

    # create an empty yaml file if it does not exist
    if not os.path.exists(path_to_conf):
        with open(path_to_conf, "w") as file:
            yaml.dump({}, file)

    # load yaml file
    with open(path_to_conf, "r") as file:
        data = yaml.load(file, Loader=yaml.FullLoader)

    # set the path
    data[attr] = path

    # save the yaml file
    with open(path_to_conf, "w") as file:
        yaml.dump(data, file)

    print(f"Set {attr} to {path} in file {path_to_conf}.")
