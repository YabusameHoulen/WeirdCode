# # General
# import matplotlib.pyplot as plt
# from astropy.io import fits
# from .GTburst_Utils import get_nearest_dets, get_ra_dec
# import pandas as pd
# from .my_general_utils import show_property, mkdir
# import gc

# # 3ML package import
# from threeML import TimeSeriesBuilder, BayesianAnalysis, DataList
# from threeML.plugins.SpectrumLike import SpectrumLike
# from threeML import display_spectrum_model_counts, plot_spectra
# from threeML import clone_model
# from astromodels import Model, PointSource
# from astromodels import functions as tml_func


# def get_light_curve(
#     GRB_name: str,
#     dets_selections: list[str],
#     chosen_ttes: list[str],
#     chosen_cspecs: list[str],
#     chosen_rsps: list[str],
#     *,
#     source_interval: str,
#     background_interval: str,
#     savedir: str,
# ) -> tuple[dict[str, TimeSeriesBuilder], list[SpectrumLike]]:
#     mkdir(f"{savedir}/{GRB_name}")
#     gbm_plugins = []
#     time_series = {}
#     for det, tte, cspec, rsp in zip(
#         dets_selections, chosen_ttes, chosen_cspecs, chosen_rsps
#     ):
#         gbmts_cspec = TimeSeriesBuilder.from_gbm_cspec_or_ctime(
#             det, cspec_or_ctime_file=cspec, rsp_file=rsp
#         )
#         gbmts_cspec.set_background_interval(*background_interval.split(","))
#         gbmts_cspec.save_background(
#             f"{savedir}/{GRB_name}/{det}_bkg.h5", overwrite=True
#         )  # save background
#         gbmts_tte = TimeSeriesBuilder.from_gbm_tte(
#             det,
#             tte_file=tte,
#             rsp_file=rsp,
#             restore_background=f"{savedir}/{GRB_name}/{det}_bkg.h5",
#         )

#         time_series[det] = gbmts_tte
#         gbmts_tte.set_active_time_interval(source_interval)

#         ### fluence_plugins setting
#         gbm_plugin = gbmts_tte.to_spectrumlike()
#         if det.startswith("b"):
#             gbm_plugin.set_active_measurements("300-35000")
#         else:
#             gbm_plugin.set_active_measurements(exclude=["0-9", "30-40", "c126-c128"])

#         gbm_plugin.rebin_on_background(0.1)
#         gbm_plugins.append(gbm_plugin)
#     return time_series, gbm_plugins


# def get_bayes_sample(
#     model: Model, data: list, sampler: str, **kwarg
# ) -> BayesianAnalysis:
#     bayes = BayesianAnalysis(model, DataList(*data))
#     bayes.set_sampler(sampler)
#     bayes.sampler.setup(**kwarg)
#     bayes.sample()
#     return bayes


# def get_bayes_blocks(
#     time_series: dict,
#     dets_selection: list[str],
#     bayes_interval: str,
#     *,
#     p0: float,
# ):
#     st_end = get_src_stend(bayes_interval)
#     for det in dets_selection:
#         time_series[det].create_time_bins(
#             start=st_end[0],
#             stop=st_end[1],
#             method="bayesblocks",
#             p0=p0,
#             use_background=True,
#         )
#         bad_bins = []
#         for i, w in enumerate(time_series[det].bins.widths):
#             if w < 5e-2:
#                 bad_bins.append(i)

#         # bad_bins = bad_bins + [
#         #     i
#         #     for i, v in enumerate(time_series[det].significance_per_interval)
#         #     if v < 15
#         # ]

#         edges = [time_series[det].bins.starts[0]]

#         for i, b in enumerate(time_series[det].bins):
#             if i not in bad_bins:
#                 edges.append(b.stop)

#         # if edges[-1] < time_series[det].bins[-1].stop:  ### 让最后的区间覆盖源区间
#         #     edges[-1] = time_series[det].bins[-1].stop

#         starts = edges[:-1]
#         stops = edges[1:]

#         time_series[det].create_time_bins(starts, stops, method="custom")


# def get_spectrum_like(time_series: dict[str, TimeSeriesBuilder], bayesdet: str):
#     "所有探头按照选择bayes块的探头分bin"
#     time_resolved_plugins = {}
#     for k, v in time_series.items():
#         v.read_bins(time_series[bayesdet])
#         time_resolved_plugins[k] = v.to_spectrumlike(from_bins=True)

#     return time_resolved_plugins


# def get_timeresolve_analysis(
#     GRB_name: str,
#     time_series: dict[str, TimeSeriesBuilder],
#     bayesdet: str,
#     fit_models: dict[str, Model],
#     *,
#     savedir,
#     **sampler_settings,
# ):

#     det_bayes_bin = time_series[bayesdet]
#     bins_divnum = len(det_bayes_bin.bins.edges) - 1
#     analysis_results = {}
#     for model_name, model in fit_models.items():
#         analysis_result = []
#         time_resolved_plugins = get_spectrum_like(time_series, bayesdet)
#         mkdir(f"{savedir}/{GRB_name}/{model_name}")
#         for interval in range(bins_divnum):
#             data_list = []
#             for k, v in time_resolved_plugins.items():
#                 pi = v[interval]
#                 if k.startswith("b"):
#                     pi.set_active_measurements("300-35000")
#                 else:
#                     pi.set_active_measurements(exclude=["0-9", "30-40", "c126-c128"])

#                 pi.rebin_on_background(1.0)
#                 data_list.append(pi)

#             timebin_range = det_bayes_bin.bins[interval].to_string()
#             print(f"{model_name} fitting in ", timebin_range)
#             gc.collect()

#             cur_model = clone_model(model)
#             bin_bayes = get_bayes_sample(cur_model, data_list, **sampler_settings)

#             ### save .fit
#             bin_bayes.results.write_to(
#                 f"{savedir}/{GRB_name}/{model_name}/{timebin_range}.fit",
#                 overwrite=True,
#             )
#             # bin_bayes.restore_median_fit()   #### MAP 估计 还是 中值估计(好像没有用)

#             bin_results = bin_bayes.results.get_data_frame()
#             pd.DataFrame.to_csv(
#                 bin_results,
#                 f"{savedir}/{GRB_name}/{model_name}/{timebin_range}.csv",
#                 mode="w",
#             )

#             bin_statistics = bin_bayes.results.get_statistic_measure_frame()
#             pd.DataFrame.to_csv(
#                 bin_statistics,
#                 f"{savedir}/{GRB_name}/{model_name}/{timebin_range}_statistics.csv",
#                 mode="w",
#             )
#             gc.collect()

#             analysis_result.append(bin_bayes)
#         analysis_results[model_name] = analysis_result
#     return analysis_results


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
#             v.restore_median_fit()
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
