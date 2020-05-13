import json

from models.ptsemseg.loader.pascal_voc_loader import pascalVOCLoader
from models.ptsemseg.loader.camvid_loader import camvidLoader
from models.ptsemseg.loader.ade20k_loader import ADE20KLoader
from models.ptsemseg.loader.mit_sceneparsing_benchmark_loader import MITSceneParsingBenchmarkLoader
from models.ptsemseg.loader.cityscapes_loader import cityscapesLoader
from models.ptsemseg.loader.nyuv2_loader import NYUv2Loader
from models.ptsemseg.loader.sunrgbd_loader import SUNRGBDLoader
from models.ptsemseg.loader.mapillary_vistas_loader import mapillaryVistasLoader


def get_loader(name):
    """get_loader

    :param name:
    """
    return {
        "pascal": pascalVOCLoader,
        "camvid": camvidLoader,
        "ade20k": ADE20KLoader,
        "mit_sceneparsing_benchmark": MITSceneParsingBenchmarkLoader,
        "cityscapes": cityscapesLoader,
        "nyuv2": NYUv2Loader,
        "sunrgbd": SUNRGBDLoader,
        "vistas": mapillaryVistasLoader,
    }[name]
