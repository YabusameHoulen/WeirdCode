# %%
# import os, sys
# ### 添加一下我的低质量脚本
# module_path = os.path.abspath(os.path.join(".."))
# if module_path not in sys.path:
#     sys.path.append(module_path)
from typing import Final, Iterable
from Myfuncs.GTburst_Utils import get_ra_dec, get_dets_angles
from Myfuncs.my_general_utils import *
from Myfuncs.fast_threeml import *

# %%
RAWDATA: Final[str] = "Raw_Data"
MYRSP_MATRIX: Final[str] = "generated_rsp_files"
SAVE_PATH: Final[str] = "3ML_GeneratedRSP2_Integrate"

grb_names = [
    "bn100619015",
    "bn121031949",
    "bn140108721",
    "bn151027166",
    "bn170514180",
    "bn171120556",
    "bn180612785",
    "bn181222279",
    "bn190419414",
]
grb_chosen_times = [
    ("0-12", "77-101"),  # "bn100619015"
    ("-5-40", "193-232"),  # "bn121031949"
    ("0-12", "78-102"),  # "bn140108721"
    ("0-38", "95-126"),  # "bn151027166"
    ("-3-30", "70-110"),  # "bn170514180"
    ("-2-4", "35-48"),  # "bn171120556"
    ("-1-26", "95-110"),  # "bn180612785"
    ("0-33", "90-102"),  # "bn181222279"
    ("-2-20", "155-220"),  # "bn190419414"
]
grb_bkg_times = [
    "-25--5,30-50,160-180",  # "bn100619015"
    "-25--10,100-120,240-260",  # "bn121031949"
    "-20--5,130-150",  # "bn140108721"
    "-25--5,40-60,150-170",  # "bn151027166"
    "-30--10,120-140",  # "bn170514180"
    "-25--5,60-80",  # "bn171120556"
    "-25--5,40-60,120-140",  # "bn180612785"
    "-30--10,50-70,160-180",  # "bn181222279"
    "-30--10,90-110,240-260",  # "bn190419414"
]

grb_files = get_GRB_files(RAWDATA, grb_names)

# %%
### bn140108721 没有rsp2文件，用rsp中的 RA DEC 来选择探头数据
bn140108_rawdata = os.listdir(joinpath(RAWDATA, "bn140108721"))
bn140108_rsps_name = sorted(list(filter(lambda x: ".rsp" in x, bn140108_rawdata)))
bn140108_rsps = [
    joinpath(RAWDATA, "bn140108721", rsp_name) for rsp_name in bn140108_rsps_name
]
bn140108_dets_angles = get_dets_angles(grb_files[2].trigdat_file, bn140108_rsps[0], 60)
grb_files[2].choose_by_dets(bn140108_dets_angles)
grb_files[2].dets_selection

### bn181222279 没有n5探头的tte文件
grb_files[7].dets_selection = {"b0":-1, "n4":-1}
# %%
for grb_name, grb_file, grb_time, grb_bkg in zip(
    grb_names, grb_files, grb_chosen_times, grb_bkg_times
):
    gc.collect()
    if grb_name == "bn140108721":
        ra, dec = get_ra_dec(bn140108_rsps[0])
    else:
        ra, dec = get_ra_dec(grb_file.rsp2_files[0])

    for pulse_time in grb_time:
        gc.collect()
        ### change the grb_file.rsp2_files to generated rsp files
        grb_path = joinpath(MYRSP_MATRIX, grb_name)
        generated_rsp2 = os.listdir(grb_path)
        chosen_rsp2s = sorted(
            [pulse_rsp2 for pulse_rsp2 in generated_rsp2 if pulse_time in pulse_rsp2]
        )
        pulse_rsp2s = [joinpath(grb_path, pulse_rsp2) for pulse_rsp2 in chosen_rsp2s]
        grb_file.rsp2_files = pulse_rsp2s

        ### 开始拟合流程
        time_series, gbm_plugins = get_timeseries(
            grb_name,
            grb_file,
            src_interval=pulse_time,
            background_interval=grb_bkg,
            savedir=SAVE_PATH,
        )
        data_dof = [
            gbm_plugin.get_number_of_data_points() for gbm_plugin in gbm_plugins
        ]
        # mkdir(f"{SAVE_PATH}/{grb_name}/dataDOF_{sum(data_dof)}_{data_dof}")
        save_light_curves(time_series, pulse_time, savedir=f"{SAVE_PATH}/{grb_name}")

        test_models = get_default_fit_models(grb_name, ra, dec)
        for model_name, models in test_models.items():
            paras_vals = None
            for i in range(2):  ######################################### 拟合两遍
                gc.collect()
                if isinstance(paras_vals, Iterable):  ########### 初始值改变在这里
                    models.set_free_parameters(paras_vals)
                pulse_bayes = get_bayes_sampling(
                    models,
                    gbm_plugins,
                    "emcee",
                    n_iterations=10000,
                    n_burn_in=2500,
                    n_walkers=20,
                )

                paras_vals = (
                    pulse_bayes.results._values
                )  ### 使用默认MAP得到的第一次的值
                # pulse_bayes.restore_median_fit() ### 这一步是否添加都没有效果?
                ### 保存结果时手动添加了的med_fit的值
                save_bayes_result(
                    pulse_bayes,
                    savedir=f"{SAVE_PATH}/{grb_name}/{model_name}/{i}_fitting",
                    fit_time=pulse_time,
                )

                save_bayes_plot(
                    pulse_bayes,
                    savedir=f"{SAVE_PATH}/{grb_name}/{model_name}/{i}_fitting",
                    fit_time=pulse_time,
                )
            free_para_num: int = len(models.free_parameters)
            mkdir(
                f"{SAVE_PATH}/{grb_name}/{model_name}/dataDOF_{sum(data_dof) - free_para_num}"
            )
#     # break ### test first burst


total_results_list: list = []

for grb_name, grb_time in zip(grb_names, grb_chosen_times):
    total_results_list.append(EMPTY_DFROW)
    burst_result = get_results_df(grb_name, grb_time, result_path=SAVE_PATH)
    total_results_list.append(burst_result)
    # break ### test first burst

total_results_df = pd.concat(total_results_list, ignore_index=True)
total_results_df.to_csv("all_generated.csv")