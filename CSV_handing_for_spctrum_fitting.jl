begin
    using CSV
    using DataFrames
    using Printf

    ### 定义一些辅助函数

    _non_text_record_dir = x -> occursin(r"^(?!.*\.txt$).*", x)
    _parameter_error_csv = x -> occursin(r"_parameter_errors.csv", x)
    _parameter_error_median_csv = x -> occursin(r"_parameter_medeq_errors.csv", x)
    _statistics_csv = x -> occursin(r"_statistics.csv", x)
    _flux_csv = x -> occursin(r"_fluxofspec.csv", x)
end


"整理按路径 ..//grb_dir//model_name//time_range_.csv"
function read_all_sp(grb_names, grb_dirs, model_name::AbstractString; para_type::Function)
    para_errs = DataFrame[]
    stats = DataFrame[]
    new_stat_name = [:AIC, :BIC, :DIC, :PDIC]
    error_counts = String[]

    for (grb_name, grb_dir) in zip(grb_names, grb_dirs)

        ### 参数 统计量 时间 都是从 all_files 中读取，确保他们之间的对应是正确的
        all_files = try
            readdir(joinpath(grb_dir, model_name), join=true)
        catch e
            @warn e
            println("examine next burst")
            push!(error_counts, grb_name)
            continue
        end

        parameter_error_csv = filter(para_type, all_files)
        flux_csv = filter(_flux_csv, all_files)
        statistics_csv = filter(_statistics_csv, all_files)

        para_df = CSV.read(parameter_error_csv, DataFrame)
        flux_df = CSV.read(flux_csv, DataFrame)
        stat_df = CSV.read(statistics_csv, DataFrame)
        ### 重新命名stat_df 四个统计量[:AIC, :BIC, :DIC, :PDIC]作为表头
        stat_df = DataFrame(reshape(stat_df[!, 2], 4, length(statistics_csv))', new_stat_name)

        stat_df.name .= grb_name
        ### flux_df 内容剪裁
        # flux_df.Column1 .= grb_name
        stat_df.flux = (Meta.parse ∘ first ∘ split).(flux_df.flux, " e")
        stat_df.flux_low = (Meta.parse ∘ first ∘ split).(flux_df.var"low bound", " e")
        stat_df.flux_high = (Meta.parse ∘ first ∘ split).(flux_df.var"hi bound", " e")
        ### time 使用正则表达式匹配路径中的正确时间
        lc_times = getproperty.(filter!(!isnothing,
                match.(r"(-?\d+\.?\d*)-(-?\d+\.?\d*)", all_files)), :match) |> unique!
        ### 之前有一些文件名没有起好,下面是个布丁
        filter!(!endswith('.'), lc_times)
        stat_df.time = lc_times
        select!(stat_df, :name, :time, Not(:name, :time))

        push!(para_errs, para_df)
        push!(stats, stat_df)
    end

    para_errs_df = reduce(vcat, para_errs)
    stats_df = reduce(vcat, stats)
    para_errs_df, stats_df, error_counts
end


const data_path = raw"1126_Integrate"
# const data_path = raw"1118_final_timeresolve"
# const data_path = raw"1120_timeresolved_analysis"
# const data_path = raw"1020_timeresolved_analysis"


begin

    grb_names = readdir(data_path)
    filter!(_non_text_record_dir, grb_names)
    grb_dirs = joinpath.(data_path, grb_names)


    ### two GRB model
    a, b, c = read_all_sp(grb_names, grb_dirs, "Band"; para_type=_parameter_error_median_csv)
    d, e, f = read_all_sp(grb_names, grb_dirs, "CPL"; para_type=_parameter_error_median_csv)
    # change_to_E_peak!(d,e,model_name="CPL") ### 3ML 中有直接的Eₚ 拟合

    CSV.write(eval(@__DIR__) * "/output_csv/band_parameters.csv", a)
    CSV.write(eval(@__DIR__) * "/output_csv/band_stats.csv", b)

    CSV.write(eval(@__DIR__) * "/output_csv/cpl_parameters.csv", d)
    CSV.write(eval(@__DIR__) * "/output_csv/cpl_stats.csv", e)

    ### with blackbody components
    a, b, c = read_all_sp(grb_names, grb_dirs, "Band_BB"; para_type=_parameter_error_median_csv)
    d, e, f = read_all_sp(grb_names, grb_dirs, "CPL_BB"; para_type=_parameter_error_median_csv)

    CSV.write(eval(@__DIR__) * "/output_csv/band_bb_parameters.csv", a)
    CSV.write(eval(@__DIR__) * "/output_csv/band_bb_stats.csv", b)

    CSV.write(eval(@__DIR__) * "/output_csv/cpl_bb_parameters.csv", d)
    CSV.write(eval(@__DIR__) * "/output_csv/cpl_bb_stats.csv", e)

end


a, b, c = read_all_sp(grb_names, grb_dirs, "BB"; para_type=_parameter_error_median_csv)
CSV.write(eval(@__DIR__) * "/output_csv/bb_parameters.csv", a)
CSV.write(eval(@__DIR__) * "/output_csv/bb_stats.csv", b)

@vsshow b

#=
### for CPL(BB)'s E_cutoff => E_peak, contains the equal_tail_interval of E_cutoff
# function change_to_E_peak!(para_errs_df, stats_df;
#     model_name,
#     E_peak_equal_tails="E_peak_equal_tails.csv")
#     ## string => expr
#     num = model_name == "CPL" ? 3 :
#           model_name == "CPL_BB" ? 5 :
#           error("not CPL model")

#     _format_interval(x) = Meta.parse.(x) .|> eval
#     E_peak_equal_tails = CSV.read(E_peak_equal_tails, DataFrame)

#     ### use @view to modify
#     index_df = @view para_errs_df[2:num:end, :]
#     xc_df = @view para_errs_df[3:num:end, :]
#     ### test whether DataFrame E_cutoff is converted
#     if size(index_df, 1) == size(xc_df, 1) &&
#        all(==("xc"), (col1_split = split.(xc_df.Column1, '.')) .|> last)

#         alpha = index_df.value
#         E_cutoff = xc_df.value
#         E_peak = DataFrame(
#             value=(alpha .+ 2) .* E_cutoff,
#             name=stats_df.name,
#             time=stats_df.time  ### coord time  para_df<===> stat_df <===> E_peak_equal_tails
#         )
#         ### join on time and name to avoid burst with same time selection
#         E_peak = leftjoin(E_peak_equal_tails, E_peak, on=[:name, :time])
#         E_peak_eqtail = _format_interval(E_peak.CPL_E_peak_interval)
#         E_peak.positive_error = getindex.(E_peak_eqtail, 2) .- E_peak.value
#         E_peak.negative_error = getindex.(E_peak_eqtail, 1) .- E_peak.value

#         ### modify the para_df
#         xc_df.value = E_peak.value
#         xc_df.positive_error = E_peak.positive_error
#         xc_df.negative_error = E_peak.negative_error
#         xc_df.error .= missing

#         ### change the name of E_cutoff => E_peak
#         for para_row in col1_split
#             para_row[5] = "xp"
#         end

#         xc_df.Column1 = join.(col1_split, '.')
#     end
#     return nothing
# end

# pattern =r"(-?\d+\.?\d*)-(-?\d+\.?\d*)"
# match(pattern,"123423.32-342234").:match
=#
# data_path = raw"C:\Users\54190\Desktop\GRB_FInal\0712_Integral"
# const data_path = raw"/home/tsubakura/Desktop/result_version_compare/1015_morning_Integrate/"