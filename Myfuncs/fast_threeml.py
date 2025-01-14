# general
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import gc
import os
import astropy.units as u

# 3ML package import
from threeML import TimeSeriesBuilder, BayesianAnalysis, DataList
from threeML.plugins.SpectrumLike import SpectrumLike
from threeML import display_spectrum_model_counts, plot_spectra
from threeML import clone_model
from astromodels import Model, PointSource
from astromodels import functions as tml_func

### some low quality code
from .GTburst_Utils import get_ra_dec, get_dets_angles
from .my_general_utils import mkdir, decode_3ml_timestr
from .my_general_utils import GRBFileDir

from typing import Iterable

### 用python 脚本跑，有的时候越跑占用内存越高最后电脑死机
### 这个绘图后端占内存小一些 ？
### 限制这个后端的chunksize防止error
plt.switch_backend("Agg")
plt.rcParams["agg.path.chunksize"] = 10000
# from threeML import threeML_config
# threeML_config["parallel"]["use_parallel"] = True


def get_timeseries(
    GRB_name: str,
    grb_file: GRBFileDir,
    *,
    src_interval: str,
    background_interval: str,
    savedir: str,
) -> tuple[dict[str, TimeSeriesBuilder], list[SpectrumLike]]:
    mkdir(f"{savedir}/{GRB_name}")
    gbm_plugins = []
    time_series = {}
    for det, tte, cspec, rsp in zip(
        grb_file.dets_selection,
        grb_file.tte_files,
        grb_file.cspec_files,
        grb_file.rsp2_files,
    ):
        ### 用cspec文件拟合背景
        gbmts_cspec = TimeSeriesBuilder.from_gbm_cspec_or_ctime(
            det, cspec_or_ctime_file=cspec, rsp_file=rsp
        )
        gbmts_cspec.set_background_interval(*background_interval.split(","))
        gbmts_cspec.save_background(
            f"{savedir}/{GRB_name}/{det}_{src_interval}_bkg.h5", overwrite=True
        )
        ### 用tte文件做谱拟合
        gbmts_tte = TimeSeriesBuilder.from_gbm_tte(
            det,
            tte_file=tte,
            rsp_file=rsp,
            restore_background=f"{savedir}/{GRB_name}/{det}_{src_interval}_bkg.h5",
        )

        time_series[det] = gbmts_tte
        gbmts_tte.set_active_time_interval(src_interval)

        ### fluence_plugins 能量设置在这里手动改，基本没问题
        gbm_plugin: SpectrumLike = gbmts_tte.to_spectrumlike()
        if det.startswith("b"):
            gbm_plugin.set_active_measurements("250-35000")
        else:
            gbm_plugin.set_active_measurements(exclude=["0-9", "30-40", "c126-c128"])

        gbm_plugin.rebin_on_background(1)
        gbm_plugins.append(gbm_plugin)
    return time_series, gbm_plugins


def get_bayes_sampling(
    model: Model, data: list, sampler: str, **kwarg
) -> BayesianAnalysis:
    bayes = BayesianAnalysis(model, DataList(*data))
    bayes.set_sampler(sampler)
    bayes.sampler.setup(**kwarg)
    bayes.sample()
    return bayes


def _cut_id_with_order(arr, threshold):
    cut_id = []
    current_sum = 0

    # Iterate through the array, combining values to ensure they meet the threshold
    for i, value in enumerate(arr):
        current_sum += value
        if current_sum >= threshold:
            cut_id.append(i)
            current_sum = 0

    if not cut_id:
        return [i]
    if cut_id[-1] != i:
        cut_id[-1] = i

    return cut_id


def Gti(test_list, sig_threshold: float = 15):
    start_indices = []  # 存储起始索引
    end_indices = []  # 存储结束索引
    in_range = False

    for index, value in enumerate(test_list):
        if value > sig_threshold:
            if not in_range:  # 如果之前不在范围内，记录起始索引
                start_indices.append(index)
                in_range = True
                continue
        else:
            if in_range:
                in_range = False  # 更新标记
                end_indices.append(index)  # 记录结束索引
    if (not end_indices) or in_range:
        end_indices.append(len(test_list) + 1)

    return start_indices, end_indices


def get_bayes_blocks(
    time_series: dict,
    dets_selection: list[str],
    bayes_interval: str,
    *,
    sig_threshold=None,
    p0: float,
) -> None:
    st_end = decode_3ml_timestr(bayes_interval)
    for det in dets_selection:
        time_series[det].create_time_bins(
            start=st_end[0],
            stop=st_end[1],
            method="bayesblocks",
            p0=p0,
            use_background=True,
        )
        bad_bins = []
        for i, w in enumerate(time_series[det].bins.widths):
            if w < 0.2:
                bad_bins.append(i)

        edges = [time_series[det].bins.starts[0]]

        for i, b in enumerate(time_series[det].bins):
            if i not in bad_bins:
                edges.append(b.stop)

        starts = edges[:-1]
        stops = edges[1:]
        time_series[det].create_time_bins(starts, stops, method="custom")

        if sig_threshold != None:
            start_indices, end_indices = Gti(
                time_series[det].significance_per_interval, sig_threshold
            )
            print(start_indices, end_indices)
            starts = []
            stops = []
            for st, ed in zip(start_indices, end_indices):
                starts += [i.start for i in time_series[det].bins[st:ed]]
                stops += [i.stop for i in time_series[det].bins[st:ed]]
            print(starts, stops)
            time_series[det].create_time_bins(starts, stops, method="custom")


def get_bayes_blocks_rebin(
    time_series: dict,
    dets_selection: list[str],
    bayes_interval: str,
    *,
    sig_threshold: float = 15,
    p0: float,
) -> None:
    st_end = decode_3ml_timestr(bayes_interval)
    for det in dets_selection:
        time_series[det].create_time_bins(
            start=st_end[0],
            stop=st_end[1],
            method="bayesblocks",
            p0=p0,
            use_background=True,
        )
        bad_bins = []
        for i, w in enumerate(time_series[det].bins.widths):
            if w < 0.2:
                bad_bins.append(i)

        edges = [time_series[det].bins.starts[0]]

        for i, b in enumerate(time_series[det].bins):
            if i not in bad_bins:
                edges.append(b.stop)

        starts = edges[:-1]
        stops = edges[1:]
        time_series[det].create_time_bins(starts, stops, method="custom")
        ### 尝试三次，不行拉倒
        test_counts = 3
        while not all(time_series[det].significance_per_interval > sig_threshold):
            if test_counts == 0:
                break
            test_counts -= 1
            cut_id = _cut_id_with_order(
                time_series[det].significance_per_interval, sig_threshold
            )
            print(cut_id)
            edges = [time_series[det].bins.starts[0]]
            for i, b in enumerate(time_series[det].bins):
                if i in cut_id:
                    edges.append(b.stop)
            starts = edges[:-1]
            stops = edges[1:]
            time_series[det].create_time_bins(starts, stops, method="custom")


def get_spectrum_like(
    time_series: dict[str, TimeSeriesBuilder], bayesdet: str
) -> dict[str, SpectrumLike]:
    "所有探头按照选择bayes块的探头分bin"
    time_resolved_plugins = {}
    for k, v in time_series.items():
        v.read_bins(time_series[bayesdet])
        time_resolved_plugins[k] = v.to_spectrumlike(from_bins=True)

    return time_resolved_plugins


def get_default_fit_models(
    GRB_name: str, RA_OBJ: float, DEC_OBJ: float
) -> dict[str, Model]:
    "克隆这个字典中的模型用来拟合分辨谱"
    fit_models = {}
    ### Band
    func = tml_func.Band()
    func.K.prior = tml_func.Log_normal(mu=0, sigma=2)
    func.alpha.prior = tml_func.Truncated_gaussian(
        lower_bound=-1.5, upper_bound=1, mu=-1, sigma=0.5
    )
    func.beta.prior = tml_func.Truncated_gaussian(
        lower_bound=-5, upper_bound=-1.8, mu=-2, sigma=0.5
    )
    func.xp.prior = tml_func.Log_uniform_prior(lower_bound=10, upper_bound=1e4)
    # func.K, func.alpha, func.beta, func.xp = [1e-2, -0.6, -2.0, 300.0]
    GRBsource = PointSource(GRB_name, RA_OBJ, DEC_OBJ, spectral_shape=func)
    fit_models["Band"] = Model(GRBsource)

    ### CPL_Ep
    func = tml_func.Cutoff_powerlaw_Ep()
    func.K.prior = tml_func.Log_uniform_prior(lower_bound=1e-10, upper_bound=1e3)
    func.index.prior = tml_func.Truncated_gaussian(
        lower_bound=-5, upper_bound=1, mu=-1, sigma=0.5
    )
    func.xp.prior = tml_func.Log_uniform_prior(lower_bound=10, upper_bound=1e4)
    # func.K, func.index, func.xc = [1e-2, -0.6, 300.0]
    GRBsource = PointSource(GRB_name, RA_OBJ, DEC_OBJ, spectral_shape=func)
    fit_models["CPL"] = Model(GRBsource)

    ## BB
    func = tml_func.Blackbody()
    func.K.prior = tml_func.Log_uniform_prior(lower_bound=1e-10, upper_bound=1e3)
    func.kT.prior = tml_func.Log_uniform_prior(lower_bound=1, upper_bound=1e3)
    GRBsource = PointSource(GRB_name, RA_OBJ, DEC_OBJ, spectral_shape=func)
    fit_models["BB"] = Model(GRBsource)

    ### Band+BB
    func = tml_func.Band() + tml_func.Blackbody()
    func.K_1.prior = tml_func.Log_normal(mu=0, sigma=2)
    func.alpha_1.prior = tml_func.Truncated_gaussian(
        lower_bound=-1.5, upper_bound=1, mu=-1, sigma=0.5
    )
    func.beta_1.prior = tml_func.Truncated_gaussian(
        lower_bound=-5, upper_bound=-1.8, mu=-2, sigma=0.5
    )
    func.xp_1.prior = tml_func.Log_uniform_prior(lower_bound=10, upper_bound=1e4)
    func.K_2.prior = tml_func.Log_uniform_prior(lower_bound=1e-10, upper_bound=1e3)
    func.kT_2.prior = tml_func.Log_uniform_prior(lower_bound=1, upper_bound=1e3)
    GRBsource = PointSource(GRB_name, RA_OBJ, DEC_OBJ, spectral_shape=func)
    fit_models["Band_BB"] = Model(GRBsource)

    ### CPLBB
    func = tml_func.Cutoff_powerlaw_Ep() + tml_func.Blackbody()
    func.K_1.prior = tml_func.Log_uniform_prior(lower_bound=1e-10, upper_bound=1e3)
    func.index_1.prior = tml_func.Truncated_gaussian(
        lower_bound=-5, upper_bound=1, mu=-1, sigma=0.5
    )
    func.xp_1.prior = tml_func.Log_uniform_prior(lower_bound=10, upper_bound=1e4)
    func.K_2.prior = tml_func.Log_uniform_prior(lower_bound=1e-10, upper_bound=1e3)
    func.kT_2.prior = tml_func.Log_uniform_prior(lower_bound=1, upper_bound=1e3)
    GRBsource = PointSource(GRB_name, RA_OBJ, DEC_OBJ, spectral_shape=func)
    fit_models["CPL_BB"] = Model(GRBsource)

    return fit_models


def get_default_fit_models_old(
    GRB_name: str, RA_OBJ: float, DEC_OBJ: float
) -> dict[str, Model]:
    "克隆这个字典中的模型用来拟合分辨谱"
    fit_models = {}

    ### Band
    func = tml_func.Band()
    func.K.prior = tml_func.Log_uniform_prior(lower_bound=1e-15, upper_bound=1e3)
    func.alpha.prior = tml_func.Truncated_gaussian(
        lower_bound=-1.5, upper_bound=1, mu=-1, sigma=0.5
    )
    func.beta.prior = tml_func.Truncated_gaussian(
        lower_bound=-5, upper_bound=-1.6, mu=-2.5, sigma=0.5
    )
    func.xp.prior = tml_func.Log_uniform_prior(lower_bound=10, upper_bound=2e3)
    # func.K, func.alpha, func.beta, func.xp = [1e-2, -0.6, -2.0, 300.0]
    GRBsource = PointSource(GRB_name, RA_OBJ, DEC_OBJ, spectral_shape=func)
    fit_models["Band"] = Model(GRBsource)

    ## CPL+BB
    func = tml_func.Cutoff_powerlaw_Ep() + tml_func.Blackbody()
    func.K_1.prior = tml_func.Log_uniform_prior(lower_bound=1e-15, upper_bound=1e3)
    func.index_1.prior = tml_func.Truncated_gaussian(
        lower_bound=-5, upper_bound=2, mu=-1, sigma=0.5
    )
    func.xp_1.prior = tml_func.Log_uniform_prior(lower_bound=10, upper_bound=2e3)
    func.K_2.prior = tml_func.Log_uniform_prior(lower_bound=1e-25, upper_bound=1)
    func.kT_2.prior = tml_func.Log_uniform_prior(lower_bound=1, upper_bound=1e3)
    GRBsource = PointSource(GRB_name, RA_OBJ, DEC_OBJ, spectral_shape=func)
    fit_models["CPL_BB"] = Model(GRBsource)

    ### CPL
    # func = tml_func.Cutoff_powerlaw()
    # func.K.prior = tml_func.Log_uniform_prior(lower_bound=1e-10, upper_bound=1e3)
    # func.index.prior = tml_func.Truncated_gaussian(
    #     lower_bound=-1.5, upper_bound=1.5, mu=-1, sigma=0.5
    # )
    # func.xc.prior = tml_func.Log_uniform_prior(lower_bound=10, upper_bound=5e4)
    # func.K, func.index, func.xc = [1e-2, -0.6, 300.0]
    # GRBsource = PointSource(GRB_name, RA_OBJ, DEC_OBJ, spectral_shape=func)
    # fit_models["CPL"] = Model(GRBsource)

    ### CPL Gaussian prior
    # func = tml_func.Cutoff_powerlaw()
    # func.K.prior = tml_func.Log_normal(mu=0, sigma=2)
    # func.index.prior = tml_func.Gaussian(mu=-1,sigma=0.5)
    # func.xc.prior = tml_func.Log_uniform_prior(lower_bound=10, upper_bound=5e4)
    # func.K, func.index, func.xc = [1e-2, -0.6, 300.0]
    # GRBsource = PointSource(GRB_name, RA_OBJ, DEC_OBJ, spectral_shape=func)
    # fit_models["CPL"] = Model(GRBsource)

    ### CPL_Ep
    func = tml_func.Cutoff_powerlaw_Ep()
    func.K.prior = tml_func.Log_uniform_prior(lower_bound=1e-15, upper_bound=1e3)
    func.index.prior = tml_func.Truncated_gaussian(
        lower_bound=-5, upper_bound=2, mu=-1, sigma=0.5
    )
    func.xp.prior = tml_func.Log_uniform_prior(lower_bound=10, upper_bound=2e3)
    # func.K, func.index, func.xc = [1e-2, -0.6, 300.0]
    GRBsource = PointSource(GRB_name, RA_OBJ, DEC_OBJ, spectral_shape=func)
    fit_models["CPL"] = Model(GRBsource)

    ## Band+BB
    func = tml_func.Band() + tml_func.Blackbody()
    func.K_1.prior = tml_func.Log_uniform_prior(lower_bound=1e-15, upper_bound=1e3)
    func.alpha_1.prior = tml_func.Truncated_gaussian(
        lower_bound=-1.5, upper_bound=1, mu=-1, sigma=0.5
    )
    func.beta_1.prior = tml_func.Truncated_gaussian(
        lower_bound=-5, upper_bound=-1.6, mu=-2.5, sigma=0.5
    )
    func.xp_1.prior = tml_func.Log_uniform_prior(lower_bound=10, upper_bound=2e3)
    func.K_2.prior = tml_func.Log_uniform_prior(lower_bound=1e-25, upper_bound=1)
    func.kT_2.prior = tml_func.Log_uniform_prior(lower_bound=1, upper_bound=1e3)
    GRBsource = PointSource(GRB_name, RA_OBJ, DEC_OBJ, spectral_shape=func)
    fit_models["Band_BB"] = Model(GRBsource)

    # func =  tml_func.Blackbody()+ tml_func.Band()
    # func.K_2.prior = tml_func.Log_uniform_prior(lower_bound=1e-10, upper_bound=1e3)
    # func.alpha_2.prior =tml_func.Truncated_gaussian(
    #     lower_bound=-1.5, upper_bound=2, mu=-1, sigma=0.5
    # )
    # func.beta_2.prior = tml_func.Truncated_gaussian(
    #     lower_bound=-3.5, upper_bound=-1.6, mu=-2, sigma=0.5
    # )
    # func.xp_2.prior = tml_func.Log_uniform_prior(lower_bound=10, upper_bound=1e4)
    # func.K_1.prior = tml_func.Log_uniform_prior(lower_bound=1e-15, upper_bound=1e3)
    # func.kT_1.prior = tml_func.Log_uniform_prior(lower_bound=10, upper_bound=1e3)
    # GRBsource = PointSource(GRB_name, RA_OBJ, DEC_OBJ, spectral_shape=func)
    # fit_models["Band_BB"] = Model(GRBsource)

    ## CPL+BB
    func = tml_func.Cutoff_powerlaw_Ep() + tml_func.Blackbody()
    func.K_1.prior = tml_func.Log_uniform_prior(lower_bound=1e-15, upper_bound=1e3)
    func.index_1.prior = tml_func.Truncated_gaussian(
        lower_bound=-5, upper_bound=2, mu=-1, sigma=0.5
    )
    func.xp_1.prior = tml_func.Log_uniform_prior(lower_bound=10, upper_bound=2e3)
    func.K_2.prior = tml_func.Log_uniform_prior(lower_bound=1e-25, upper_bound=1)
    func.kT_2.prior = tml_func.Log_uniform_prior(lower_bound=1, upper_bound=1e3)
    GRBsource = PointSource(GRB_name, RA_OBJ, DEC_OBJ, spectral_shape=func)
    fit_models["CPL_BB"] = Model(GRBsource)

    # func = tml_func.Blackbody()
    # func.K.prior = tml_func.Log_uniform_prior(lower_bound=1e-25, upper_bound=1e3)
    # func.kT.prior = tml_func.Log_uniform_prior(lower_bound=1, upper_bound=1e3)
    # GRBsource = PointSource(GRB_name, RA_OBJ, DEC_OBJ, spectral_shape=func)
    # fit_models["BB"] = Model(GRBsource)

    # func = tml_func.Cutoff_powerlaw() + tml_func.Blackbody()
    # func.K_1.prior = tml_func.Log_uniform_prior(lower_bound=1e-10, upper_bound=1e3)
    # func.index_1.prior = tml_func.Truncated_gaussian(
    #     lower_bound=-2, upper_bound=2, mu=-1, sigma=0.5
    # )
    # func.xc_1.prior = tml_func.Log_uniform_prior(lower_bound=10, upper_bound=5e4)
    # func.K_2.prior = tml_func.Log_uniform_prior(lower_bound=1e-15, upper_bound=1e3)
    # func.kT_2.prior = tml_func.Log_uniform_prior(lower_bound=10, upper_bound=1e3)
    # GRBsource = PointSource(GRB_name, RA_OBJ, DEC_OBJ, spectral_shape=func)
    # fit_models["CPL_BB"] = Model(GRBsource)
    return fit_models


def get_timeresolve_analysis(
    GRB_name: str,
    time_series: dict[str, TimeSeriesBuilder],
    bayesdet: str,
    fit_models: dict[str, Model],
    *,
    savedir: str,
    **sampler_settings,
):

    det_bayes_bin = time_series[bayesdet]
    bins_divnum = len(det_bayes_bin.bins)
    analysis_results = {}
    for model_name, model in fit_models.items():
        analysis_result = []
        time_resolved_plugins = get_spectrum_like(time_series, bayesdet)
        mkdir(f"{savedir}/{GRB_name}/{model_name}")
        for interval in range(bins_divnum):
            data_list = []
            for k, v in time_resolved_plugins.items():
                pi = v[interval]
                if k.startswith("b"):
                    pi.set_active_measurements("300-35000")
                else:
                    pi.set_active_measurements(exclude=["0-9", "30-40", "c126-c128"])

                pi.rebin_on_background(1.0)
                data_list.append(pi)

            timebin_range = det_bayes_bin.bins[interval].to_string()
            print(f"{model_name} fitting in ", timebin_range)
            gc.collect()

            cur_model = clone_model(model)
            bin_bayes = get_bayes_sampling(cur_model, data_list, **sampler_settings)

            save_bayes_result(
                bin_bayes,
                savedir="{savedir}/{GRB_name}/{model_name}",
                fit_time=timebin_range,
            )

            ### save .fit
            # bin_bayes.results.write_to(
            #     f"{savedir}/{GRB_name}/{model_name}/{timebin_range}.fit",
            #     overwrite=True,
            # )
            # # bin_bayes.restore_median_fit()   #### MAP 估计 还是 中值估计(好像没有用)

            # bin_results = bin_bayes.results.get_data_frame()
            # pd.DataFrame.to_csv(
            #     bin_results,
            #     f"{savedir}/{GRB_name}/{model_name}/{timebin_range}.csv",
            #     mode="w",
            # )

            # bin_statistics = bin_bayes.results.get_statistic_measure_frame()
            # pd.DataFrame.to_csv(
            #     bin_statistics,
            #     f"{savedir}/{GRB_name}/{model_name}/{timebin_range}_statistics.csv",
            #     mode="w",
            # )

            gc.collect()
            analysis_result.append(bin_bayes)
        analysis_results[model_name] = analysis_result
    return analysis_results


def save_light_curves(
    time_series: dict[str, TimeSeriesBuilder],
    src_interval: str,
    padding: int = 5,
    dt: float = 1.0,
    use_binner: bool = False,
    *,
    savedir: str,
) -> None:
    mkdir(savedir)
    view_range = decode_3ml_timestr(src_interval)

    view_range[0] = view_range[0] - padding
    view_range[1] = view_range[1] + padding

    for k, v in time_series.items():
        print(f"{k}'s light curve: \n")
        if use_binner and len(v.bins) > 1:
            fig = v.view_lightcurve(*view_range, dt=dt, use_binner=use_binner)
            fig.savefig(f"{savedir}/{k}_{src_interval}_light_curve.png")
        else:
            fig = v.view_lightcurve(*view_range, dt=dt, use_binner=False)
            fig.savefig(f"{savedir}/{k}_{src_interval}_light_curve.png")
        plt.close()


def save_bayes_result(bayes: BayesianAnalysis, *, savedir: str, fit_time: str) -> None:
    ### save chains
    mkdir(savedir)
    # np.save(f"{savedir}/samples_{fit_time}", bayes.results.samples) ### 原始样本
    # bin_bayes.restore_median_fit()   #### MAP 估计 还是 中值估计(好像没有用)

    pulse_results = bayes.results.get_data_frame()
    pd.DataFrame.to_csv(
        pulse_results,
        f"{savedir}/{fit_time}_parameter_errors.csv",
        mode="w",
    )

    bin_statistics = bayes.results.get_statistic_measure_frame()
    pd.DataFrame.to_csv(
        bin_statistics,
        f"{savedir}/{fit_time}_statistics_measure.csv",
        mode="w",
    )

    bin_statistics = bayes.results.get_statistic_frame()
    pd.DataFrame.to_csv(
        bin_statistics,
        f"{savedir}/{fit_time}_statistics.csv",
        mode="w",
    )

    ### median fit dataframe
    new_frame = pulse_results.copy(deep=True)
    for i, para_path in enumerate(pulse_results.index.values):
        random_var = bayes.results.get_variates(para_path)
        eq_tail = random_var.equal_tail_interval()
        nega_err, posi_err = (
            eq_tail[0] - random_var.median,
            eq_tail[1] - random_var.median,
        )
        new_error = (np.abs(nega_err) + posi_err) / 2.0
        new_frame.iloc[i, :4] = [random_var.median, nega_err, posi_err, new_error]

    pd.DataFrame.to_csv(
        new_frame,
        f"{savedir}/{fit_time}_parameter_medeq_errors.csv",
        mode="w",
    )

    flux_of_spec = bayes.results.get_flux(
        ene_min=10 * u.keV,
        ene_max=40 * u.MeV,
        flux_unit="erg/cm**2/s",
        use_components=True,
    )

    pd.DataFrame.to_csv(
        flux_of_spec,
        f"{savedir}/{fit_time}_fluxofspec.csv",
        mode="w",
    )


def save_bayes_plot(bayes: BayesianAnalysis, *, savedir: str, fit_time: str) -> None:
    mkdir(savedir)
    fig = display_spectrum_model_counts(bayes)
    fig.axes[0].set_xlim(8, 50000)
    fig.axes[1].set_xlim(8, 50000)
    fig.axes[0].set_ylim(1e-7, 500)
    fig.savefig(f"{savedir}/{fit_time}_spectrum_fit.png", dpi=600)
    plt.close()

    ### save trace plot
    bin_trace = bayes.plot_chains(thin=20)
    for i in range(len(bin_trace)):
        bin_trace[i].savefig(f"{savedir}/{fit_time}_tp{i}.png", dpi=600)
        plt.close()

    ### save fluence figure
    bin_flufig = plot_spectra(
        bayes.results,
        flux_unit="erg2/(cm2 s keV)",
        equal_tailed=True,
        ene_min=8.0,
        ene_max=40000.0,
        use_components=True,
    )
    bin_flufig.axes[0].set_ylim(ymax=10e-15,ymin=1e-19)
    bin_flufig.savefig(f"{savedir}/{fit_time}_flu.png", dpi=600)
    plt.close()

    ### save useless pair plots
    bin_cornor = bayes.results.corner_plot(fill_contours=True)
    bin_cornor.savefig(f"{savedir}/{fit_time}_cor.png", dpi=600)
    plt.close()


COLS = [
    "GRB_name",
    "fit_num",
    "Time",
    "model_name",
    "dof",
    "AIC",
    "BIC",
    "DIC",
    "PDIC",
    "log_posterior",
    "para_names",
    "para_values",
    "para_errors_negative",
    "para_errors_positive",
    "flux_main",
    "flux_add",
]
EMPTY_DFROW = pd.DataFrame([{col: None for col in COLS}])


def get_results_df(grb_name: str, fit_times: Iterable[str], *, result_path: str):

    total_results_list: list = []

    model_names = [
        item
        for item in os.listdir(f"{result_path}/{grb_name}")
        if os.path.isdir(os.path.join(f"{result_path}/{grb_name}", item))
    ]
    model_names = sorted(model_names)

    for fit_time in fit_times:
        total_results_list.append(EMPTY_DFROW)  ### 加入空行分割
        for model_name in model_names:
            here_dirs = os.listdir(f"{result_path}/{grb_name}/{model_name}/")
            fit_dof = list(filter(lambda x: x.startswith("d"), here_dirs))[0].split(
                "_"
            )[1]
            fit_nums = sorted(list(filter(lambda x: not x.startswith("d"), here_dirs)))

            for fit_num in fit_nums:
                # 读取文件
                paras = pd.read_csv(
                    f"{result_path}/{grb_name}/{model_name}/{fit_num}/{fit_time}_parameter_medeq_errors.csv"
                )
                loglike = pd.read_csv(
                    f"{result_path}/{grb_name}/{model_name}/{fit_num}/{fit_time}_statistics.csv"
                )
                statistics = pd.read_csv(
                    f"{result_path}/{grb_name}/{model_name}/{fit_num}/{fit_time}_statistics_measure.csv"
                )
                flux = pd.read_csv(
                    f"{result_path}/{grb_name}/{model_name}/{fit_num}/{fit_time}_fluxofspec.csv"
                )

                para_names = paras.iloc[:, 0].tolist()
                loglike.set_index("Unnamed: 0", inplace=True)
                statistics.set_index("Unnamed: 0", inplace=True)
                model_flux: dict = {"flux_main": flux.flux[0], "flux_add": None}
                if "_" in model_name:
                    model_flux["flux_add"] = flux.flux[1]

                this_result = pd.DataFrame(
                    {
                        "GRB_name": grb_name,
                        "Time": fit_time,
                        "fit_num": fit_num,
                        "model_name": model_name,
                        "dof": fit_dof,
                        **statistics["statistical measures"].to_dict(),
                        "log_posterior": loglike["-log(posterior)"]["total"],
                        "para_names": [para_names],
                        "para_values": [paras.value.values],
                        "para_errors_negative": [paras.negative_error.values],
                        "para_errors_positive": [paras.positive_error.values],
                        **model_flux,
                    }
                )
                # 整合文件
                total_results_list.append(this_result)

    total_results_df = pd.concat(total_results_list, ignore_index=True)
    return total_results_df


# def get_status_plots(
#     GRB_name: str,
#     analysis_results: dict,
#     det_bayes_timeseries: TimeSeriesBuilder,
#     *,
#     savedir: str,
# ):
#     mkdir(f"{savedir}/{GRB_name}/")
#     timebin = det_bayes_timeseries.bins.to_string().split(",")
#     for k, analysis in analysis_results.items():
#         mkdir(f"{savedir}/{GRB_name}/{k}")
#         for timebin_range, v in zip(timebin, analysis):
#             # v.restore_median_fit()
#             fig = display_spectrum_model_counts(v)
#             fig.axes[0].set_ylim(1e-7, 100)
#             fig.savefig(f"{savedir}/{GRB_name}/{k}/{timebin_range}.png", dpi=600)
#             plt.close()

#             ### save trace plot
#             bin_trace = v.plot_chains()
#             for i in range(len(bin_trace)):
#                 bin_trace[i].savefig(
#                     f"{savedir}/{GRB_name}/{k}/{timebin_range}_tp{i}.png", dpi=600
#                 )
#                 plt.close()

#             ### save fluence figure
#             bin_flufig = plot_spectra(v.results, flux_unit="erg2/(cm2 s keV)")
#             bin_flufig.savefig(
#                 f"{savedir}/{GRB_name}/{k}/{timebin_range}_flu.png", dpi=600
#             )
#             plt.close()

#             ### save useless pair plots
#             bin_cornor = v.results.corner_plot()
#             bin_cornor.savefig(
#                 f"{savedir}/{GRB_name}/{k}/{timebin_range}_cor.png", dpi=600
#             )
#             plt.close()

#         fig = plot_spectra(
#             *[a.results for a in analysis[::1]],
#             flux_unit="erg2/(cm2 s keV)",
#             fit_cmap="viridis",
#             contour_cmap="viridis",
#             contour_style_kwargs=dict(alpha=0.1),
#         )
#         fig.savefig(f"{savedir}/{GRB_name}/{k}/fluence_all.png", dpi=600)
#         plt.close()
