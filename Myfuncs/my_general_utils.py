import os
from dataclasses import dataclass
from typing import Optional
from pprint import pformat
from .GTburst_Utils import get_dets_angles


@dataclass
class GRBFileDir:

    trigdat_file: str
    tte_files: list[str]
    cspec_files: list[str]
    rsp2_files: list[str]
    dets_selection: Optional[dict[str,float]] = None
    # dets_selection: list[str]|None = None ### is supportted in python above 3.10

    def __repr__(self) -> str:
        return f"GRBFIleDir(trigdat_file:{self.trigdat_file}\ntte_files:{pformat(self.tte_files)}\ncspec_files:{pformat(self.cspec_files)}\nrsp2_files:{pformat(self.rsp2_files)})"

    def choose_by_dets(self, chosen_dets: dict):
        """
        select dets in chosen_dets and perform the compatible test
        if chosen files is incompatible, return false
        """

        print("perform the dets selection, filter the unchosen file")
        self.tte_files = [
            tte_file
            for tte_file in self.tte_files
            if os.path.split(tte_file)[1][8:10] in chosen_dets.keys()
        ]

        self.cspec_files = [
            cspec_file
            for cspec_file in self.cspec_files
            if os.path.split(cspec_file)[1][10:12] in chosen_dets.keys()
        ]

        self.rsp2_files = [
            rsp2_file
            for rsp2_file in self.rsp2_files
            if os.path.split(rsp2_file)[1][10:12] in chosen_dets.keys()
        ]
        self.compatible_test()
        self.dets_selection = dict(sorted(chosen_dets.items()))

    def compatible_test(self) -> bool:
        if not self.trigdat_file:
            print("the trigdat_file is missing")
            return False
        if (tte_len := len(self.tte_files)) == 0:
            print("the tte_files is missing")
            return False
        if (cspec_len := len(self.cspec_files)) == 0:
            print("the cspec_files is missing")
            return False
        if (rsp2_len := len(self.rsp2_files)) == 0:
            print("the rsp2_files is missing")
            return False

        if tte_len == cspec_len == rsp2_len:
            print("the GRB files are compatible")
            return True
        else:
            print("the GRB files are incompatible")
            print(
                f"tte_files: {tte_len}\t cspec_files: {cspec_len}\t rsp2_files: {rsp2_len}"
            )
            return False


def show_property(cls) -> None:
    "I like use this function in ipynb ..."
    [print(attr) for attr in dir(cls) if not attr.startswith("_")]


def mkdir(path: str) -> None:
    "a wrapper to os.makedirs, check whether there is path folder"
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
        print("---  new folder  ---")
    else:
        print("---  There is a folder with the same name!  ---")


def joinpath(path1, *path2) -> str:
    "for julia users like me ..."
    return os.path.join(path1, *path2)


def decode_3ml_timestr(source_interval: str, splitchar="-") -> list[float]:
    """convert "-0.2-20.0" to [-0.2,20.0]"""
    src_str = source_interval.split(splitchar)
    src_range = []
    for i, v in enumerate(src_str):
        if not v == "":
            if i > 0 and src_str[i - 1] == "":
                src_range.append(-eval(v))
            else:
                src_range.append(eval(v))

    if src_range[1] < src_range[0]:
        raise Exception(f"wrong src_range selection: {src_range}")
    return src_range


def get_GRBFileDir(GRB_path: str):
    """
    GRB_path : fold directory with trig tte cspec rsp2
    return file paths are sorted
    """
    burst_file = os.listdir(GRB_path)
    trigdat_file = list(filter(lambda x: "_trigdat_" in x, burst_file))[0]
    tte_files = sorted(list(filter(lambda x: "_tte_" in x, burst_file)))
    cspec_files = sorted(list(filter(lambda x: ".pha" in x and "_cspec_" in x, burst_file)))
    rsp2_files = sorted(list(filter(lambda x: ".rsp2" in x and "_cspec_" in x, burst_file)))

    ### the full path of files
    trigdat_file = joinpath(GRB_path, trigdat_file)
    tte_files = [joinpath(GRB_path, file) for file in tte_files]
    cspec_files = [joinpath(GRB_path, file) for file in cspec_files]
    rsp2_files = [joinpath(GRB_path, file) for file in rsp2_files]

    return GRBFileDir(trigdat_file, tte_files, cspec_files, rsp2_files)


def get_GRB_files(
    Data_Path: str, grb_names: list[str], chosen_angle=50, nai_num=3
) -> list[GRBFileDir]:
    grb_files = []
    for grb_name in grb_names:
        grb_path = joinpath(Data_Path, grb_name)
        grb_file = get_GRBFileDir(grb_path)
        if grb_file.rsp2_files:
            dets_angles = get_dets_angles(
                grb_file.trigdat_file, grb_file.rsp2_files[0], chosen_angle, nai_num
            )
            # print(dets_angles)
            grb_file.choose_by_dets(dets_angles)
        else:
            print(f"\033[31mthe GRBfile {grb_name} does not have rsp2 files !!!\033[0m")
        grb_files.append(grb_file)
        # print()
    return grb_files
