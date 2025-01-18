# GBM
from gbm.data import RSP, Cspec, TTE, GbmDetectorCollection, PHA  ### 数据类
from gbm.binning.unbinned import bin_by_time  # bin_by_snr,bin_by_edges
from gbm.plot import Lightcurve, Spectrum, ModelFit  ### 画图类
from gbm.background import BackgroundFitter  ### 背景拟合类
from gbm.background.binned import Polynomial
from gbm.spectra.fitting import SpectralFitterPgstat  ### 能谱拟合类
import gbm.spectra.functions as gbm_func

# Here
from Myfuncs.my_general_utils import *
import pandas as pd
from typing import Final
import gc
import matplotlib.pyplot as plt

plt.switch_backend("Agg")
plt.rcParams["agg.path.chunksize"] = 10000
# %%
################################################################### 初值设置
MYRSP_MATRIX: Final[str] = "generated_rsp_files"
RAWDATA: Final[str] = "Raw_Data"
SAVEPATH: Final[str] = "GBMFIT_Origin_Final"
niteration = 100000
numerical_methods = "TNC"  # "SLSQP"
erange_nai = (10.0, 900.0)  # keV
erange_bgo = (300.0, 35000.0)


# %%
class FitResult:
    model_symbol: dict[str, list[str]] = {
        "Band": ["A", "Ep", "alpha", "Beta"],
        "Comptonized": ["A", "Ec", "alpha"],
        "Band+BlackBody": ["A", "Ep", "alpha", "Beta", "Ab", "kT"],
        "Comptonized+BlackBody": ["A", "Ec", "alpha", "Ab", "kT"],
    }

    def __init__(self, specfitter: SpectralFitterPgstat, model_name, cl: float = 0.683):
        self.model_name: str = model_name
        self.fit_success: bool = specfitter.success
        self.message: str = specfitter.message
        self.parameters: list[float] = specfitter.parameters
        self.asyerrors: list[list[float]] = specfitter.asymmetric_errors(cl)
        self.statistics: float = specfitter.statistic
        self.dof: float = specfitter.dof

    def __repr__(self):
        indent = " " * 4
        cls = self.__class__
        cls_name = cls.__name__
        res = [f"{self.model_name} {cls_name}("]
        res.append(f"{indent}fit success: {self.fit_success}")
        res.append(f"{indent}message: {self.message}")
        if self.model_name in cls.model_symbol.keys():
            for symbol, val, asyerr in zip(
                cls.model_symbol[self.model_name], self.parameters, self.asyerrors
            ):
                res.append(
                    f"{indent}{symbol.ljust(10)} {val:.4f} -{asyerr[0]:.4f} + {asyerr[1]:.4f}"
                )
        res.append(f"{indent}Pgstat: {self.statistics}")
        res.append(f"{indent}dof: {self.dof}")
        res.append(")")
        return "\n".join(res)

    @staticmethod  ### 相关的类函数
    def parse_fit_results(grb_name, grb_time, file_path):
        with open(file_path, "r") as file:
            lines = file.readlines()

        data = []
        current_model = {}

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if line.endswith("FitResult("):
                if current_model:
                    data.append(current_model)
                current_model = current_model = {
                    "Name": grb_name,
                    "Time": grb_time,
                    "Model": line.split(" ")[0],
                }
            elif line.startswith("fit success:"):
                current_model["Fit Success"] = line.split(": ")[1]
            elif line.startswith("message:"):
                current_model["Reason"] = line.split(": ")[1]
            elif line.startswith("Pgstat:"):
                current_model["Pgstat"] = float(line.split(": ")[1])
            elif line.startswith("dof:"):
                current_model["DOF"] = int(line.split(": ")[1])
            elif line[0].isalpha():
                key, values = line.split(maxsplit=1)
                current_model[key] = values

        if current_model:
            data.append(current_model)

        return data


# %%
################################################################### 对一些暴特殊设置
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

normal_bursts = pd.read_csv("chosen_GRB_informaions.csv")
normal_bursts = (
    normal_bursts[~normal_bursts.name.isin(grb_names)]
    .reset_index()
    .drop(columns=["index"])
)

grb_files = get_GRB_files(RAWDATA, normal_bursts.name)

### 081126899 选前三个探头(三个响应文件)
### 151231443 选择两个探头
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
###################################################################### 开始循环
for grb_name, grb_file, grb_time, grb_bkg in zip(
    normal_bursts.name,
    grb_files,
    normal_bursts.grb_chosen_times,
    normal_bursts.background_interval,
):
    gc.collect()
    ### TTE ---> Binned Data
    tte_list = [TTE.open(tte) for tte in grb_file.tte_files]
    rsp_list = [RSP.open(rsp) for rsp in grb_file.rsp2_files]
    data_size = min(len(rsp_list), len(tte_list))

    ttes = GbmDetectorCollection.from_list(tte_list[:data_size])
    phaiis = GbmDetectorCollection.from_list(
        [tte.to_phaii(bin_by_time, 0.64, time_ref=0.0) for tte in ttes]
    )
    print(grb_name, ":", phaiis.detector())

    ### RSP
    rsps = GbmDetectorCollection.from_list(rsp_list[:data_size])

    ### 背景拟合
    bkgd_range = list(map(decode_3ml_timestr, grb_bkg.split(",")))
    print("Choose background: ", bkgd_range)
    backfitters = [
        BackgroundFitter.from_phaii(phaii, Polynomial, time_ranges=bkgd_range)
        for phaii in phaiis
    ]
    backfitters = GbmDetectorCollection.from_list(backfitters, dets=ttes.detector())
    backfitters.fit(order=1)

    bkgds = backfitters.interpolate_bins(
        phaiis.data()[0].tstart, phaiis.data()[0].tstop
    )
    bkgds = GbmDetectorCollection.from_list(bkgds, dets=ttes.detector())

    ### 30keV ~ 40KeV 的channel mask
    chan_masks = [(i.emin < 30) | (i.emax > 40) for i in phaiis.data()]
    # chan_masks = [i.emin > -1 for i in phaiis.data()]

    ### 对两个峰分别拟合
    for src_range in list(map(decode_3ml_timestr, eval(grb_time))):
        gc.collect()
        print("src_range: ", src_range)
        # 光变
        data_lcs = phaiis.to_lightcurve(
            nai_kwargs={"energy_range": erange_nai},
            bgo_kwargs={"energy_range": erange_bgo},
        )
        # the energy-integrated background
        bkgd_lcs = bkgds.integrate_energy(nai_args=erange_nai, bgo_args=erange_bgo)
        # 源时间选择
        src_lcs = phaiis.to_lightcurve(
            time_range=src_range,
            nai_kwargs={"energy_range": erange_nai},
            bgo_kwargs={"energy_range": erange_bgo},
        )

        # 计数谱
        data_specs = phaiis.to_spectrum(time_range=src_range)
        # the time-integrated background
        bkgd_specs = bkgds.integrate_time(*src_range)
        # 能量选择
        src_specs = phaiis.to_spectrum(
            time_range=src_range,
            nai_kwargs={"energy_range": erange_nai},
            bgo_kwargs={"energy_range": erange_bgo},
        )

        ### 检查图像
        mkdir(f"{SAVEPATH}/{grb_name}")
        lcplots = [
            Lightcurve(data=data_lc, background=bkgd_lc)
            for data_lc, bkgd_lc in zip(data_lcs, bkgd_lcs)
        ]
        for det, src_lc, lcplot in zip(bkgds._dets, src_lcs, lcplots):
            lcplot.add_selection(src_lc)
            lcplot.fig.savefig(f"{SAVEPATH}/{grb_name}/{det}_{src_range}_lc.png")

        specplots = [
            Spectrum(data=data_spec, background=bkgd_spec)
            for data_spec, bkgd_spec in zip(data_specs, bkgd_specs)
        ]
        for det, src_spec, specplot in zip(bkgds._dets, src_specs, specplots):
            specplot.add_selection(src_spec)
            specplot.fig.savefig(f"{SAVEPATH}/{grb_name}/{det}_{src_range}_spec.png")

        ### 能谱拟合
        phas = phaiis.to_pha(
            time_ranges=src_range,
            nai_kwargs={"energy_range": erange_nai},
            bgo_kwargs={"energy_range": erange_bgo},
        )

        # interpolate response files to get DRMs at center of the source window
        # rsps_interp = [rsp.interpolate(pha.tcent) for rsp, pha in zip(rsps, phas)]

        # print([chan_mask for chan_mask in chan_masks])
        steds = [(pha.valid_channels[0], pha.valid_channels[-1]) for pha in phas]
        for chan_mask, sted in zip(chan_masks, steds):
            chan_mask[: sted[0]] = False
            chan_mask[sted[1] + 1 :] = False

        # print([pha.valid_channels for pha in phas])
        # print(chan_masks)

        models = {
            "Band": gbm_func.Band(),
            "BandBB": gbm_func.Band() + gbm_func.BlackBody(),
            "CPL": gbm_func.Comptonized(),
            "CPLBB": gbm_func.Comptonized() + gbm_func.BlackBody(),
        }
        for model_str, model in models.items():
            gc.collect()
            # 初始化
            specfitter = SpectralFitterPgstat(
                phas,
                bkgds.to_list(),
                rsps.to_list(),
                # method="TNC",
                method=numerical_methods,
                channel_masks=chan_masks,
            )

            specfitter.fit(model, options={"maxfun": niteration})

            with open(f"{SAVEPATH}/{grb_name}/{src_range}_fit.txt", "a") as f:
                f.write(str(FitResult(specfitter, model.name)))
                f.write("\n\n")

            data = FitResult.parse_fit_results(
                grb_name=grb_name,
                grb_time=src_range,
                file_path=f"{SAVEPATH}/{grb_name}/{src_range}_fit.txt",
            )
            data_df = pd.DataFrame(data)
            data_df.to_csv(f"{SAVEPATH}/{grb_name}/{src_range}_fit.csv", index=False)

            try:
                modelplot = ModelFit(fitter=specfitter)
                modelplot.ax.set_ylim(1e-7, 500)
                modelplot.fig.savefig(
                    f"{SAVEPATH}/{grb_name}/{src_range}_{model_str}_fits.png"
                )
                modelplot.energy_spectrum(num_samples=200)
                modelplot.ax.set_ylim(1e-7, 10)
                modelplot.fig.savefig(
                    f"{SAVEPATH}/{grb_name}/{src_range}_{model_str}_flu.png"
                )
            except:
                print(f"{grb_name}------{src_range}-------ploterror")
                continue

            # with open(f"{SAVEPATH}/{grb_name}/{src_range}_fit.txt", "a") as f:
            #     f.write(f"{model_str}\nFit Success: {specfitter.success}\n")
            #     f.write(specfitter.message)
            #     f.write("\nParameters: {}".format(specfitter.parameters))
            #     f.write(
            #         "\n68% Asymm. Errors:\n {}".format(
            #             specfitter.asymmetric_errors(cl=0.683)
            #         )
            #     )
            #     f.write(
            #         "\nPgstat/DoF: {}/{}\n\n".format(
            #             specfitter.statistic, specfitter.dof
            #         )
            #     )

            # try:
            #     modelplot_1 = ModelFit(fitter=specfitter)
            #     modelplot_1.ax.set_ylim(1e-7, 500)
            #     modelplot_1.fig.savefig(
            #         f"{SAVEPATH}/{grb_name}/{src_range}_{model_str}_fits.png"
            #     )
            #     modelplot_2 = ModelFit(fitter=specfitter, view="nufnu")
            #     modelplot_2.ax.set_ylim(1e-7, 500)
            #     modelplot_2.fig.savefig(
            #         f"{SAVEPATH}/{grb_name}/{src_range}_{model_str}_nufnu.png"
            #     )
            # except:
            #     print(f"{grb_name}------{src_range}-------ploterror")
            #     continue

# %%
results_csv = "GBM_all_origin.csv"
total_results_list: list = []
COLS = [
    "Name",
    "Time",
    "Model",
    "Fit Success",
    "Reason",
    "Pgstat",
    "dof",
    "A",
    "Ep",
    "alpha",
    "Beta",
    "Ab",
    "kT",
    "Ec",
]
EMPTY_DFROW = pd.DataFrame([{col: None for col in COLS}])
for grb_name, grb_times in zip(normal_bursts.name, normal_bursts.grb_chosen_times):
    for grb_time in eval(grb_times):
        time_select_region = decode_3ml_timestr(grb_time)
        total_results_list.append(EMPTY_DFROW)
        burst_result = pd.read_csv(
            f"{SAVEPATH}/{grb_name}/{time_select_region}_fit.csv"
        )
        total_results_list.append(burst_result)
        # break ### test first burst

total_results_df = pd.concat(total_results_list)
total_results_df.to_csv(results_csv)
print(f"----{results_csv:^10}----")

# %%
