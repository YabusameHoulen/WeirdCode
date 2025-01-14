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
### change NaI/BGO range manually in package...
RAWDATA: Final[str] = "Raw_Data"
# MYRSP_MATRIX: Final[str]  = "generated_rsp_files"
SAVE_PATH: Final[str] = "3ML_Origin_Integrate"
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
    ### GRB listed above are fitted with self-generated rsp2 file.
]

# %%
normal_bursts = pd.read_csv("chosen_GRB_informaions.csv")
normal_bursts = (
    normal_bursts[~normal_bursts.name.isin(grb_names)]
    .reset_index()
    .drop(columns=["index"])
)

grb_files = get_GRB_files(RAWDATA, normal_bursts.name)

# %%

### 正常暴bn151231443 探头选择一个较近的NaI
grb_files[11].dets_selection = {"b1": -1, "n8": -1}
grb_files[11].tte_files = [
    "Raw_Data/bn151231443/glg_tte_b1_bn151231443_v00.fit",
    "Raw_Data/bn151231443/glg_tte_n8_bn151231443_v00.fit",
]
grb_files[11].cspec_files = [
    "Raw_Data/bn151231443/glg_cspec_b1_bn151231443_v00.pha",
    "Raw_Data/bn151231443/glg_cspec_n8_bn151231443_v00.pha",
]
grb_files[11].rsp2_files = [
    "Raw_Data/bn151231443/glg_cspec_b1_bn151231443_v00.rsp2",
    "Raw_Data/bn151231443/glg_cspec_n8_bn151231443_v00.rsp2",
]
# %%
### using normal burst table
for grb_name, grb_file, grb_time, grb_bkg in zip(
    normal_bursts.name,
    grb_files,
    normal_bursts.grb_chosen_times,
    normal_bursts.background_interval,
):
    gc.collect()
    # if grb_name == "bn140108721":
    #     ra, dec = get_ra_dec(bn140108_rsps[0])
    # else:
    ra, dec = get_ra_dec(grb_file.rsp2_files[0])

    for pulse_time in eval(grb_time):
        gc.collect()

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

    # break  ### test first burst


total_results_list: list = []

for grb_name, grb_time in zip(normal_bursts.name, normal_bursts.grb_chosen_times):
    total_results_list.append(EMPTY_DFROW)
    burst_result = get_results_df(grb_name, eval(grb_time), result_path=SAVE_PATH)
    total_results_list.append(burst_result)
    # break ### test first burst

total_results_df = pd.concat(total_results_list, ignore_index=True)
total_results_df.to_csv("all_origin.csv")
