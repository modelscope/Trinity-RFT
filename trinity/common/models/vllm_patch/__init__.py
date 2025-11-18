import vllm
from packaging.version import InvalidVersion
from packaging.version import parse as parse_version


def get_vllm_version():
    try:
        vllm_version = parse_version(vllm.__version__)
    except InvalidVersion:
        # for self-compiled vllm,
        # we cannot parse the version, trait it as the lowest version we support
        vllm_version = parse_version("0.8.5")
    return vllm_version
