import marimo

__generated_with = "0.17.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import math
    import numpy as np
    import os
    import datetime
    import random
    import calendar
    import warnings
    from scipy.special import comb
    from scipy.stats import gaussian_kde
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    import matplotlib.dates as mdates
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import matplotlib
    from matplotlib.ticker import MaxNLocator
    import matplotlib.colors as colors
    import seaborn as sns
    from aquarel import load_theme
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import pickle
    from scipy.stats import gaussian_kde

    theme = load_theme("arctic_light")
    theme.set_transforms(trim=True)
    theme.apply()
    plt.rcParams.update(
        {
            'font.size': 14,
            'axes.titleweight': 'normal',
            'figure.titleweight': 'normal'
        }
    )

    # date format
    my_date_format_str = '%b-%d'
    my_date_format = mdates.DateFormatter(my_date_format_str)
    return (
        calendar,
        ccrs,
        cfeature,
        colors,
        comb,
        datetime,
        inset_axes,
        matplotlib,
        mo,
        mpl,
        np,
        os,
        pd,
        pickle,
        plt,
        sns,
        warnings,
    )


@app.cell
def _(plt, sns):
    theme_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    sns.color_palette(theme_colors)
    return (theme_colors,)


@app.cell
def _(sns):
    my_colors = sns.color_palette("Set2")
    my_colors
    return (my_colors,)


@app.cell
def _(my_colors, theme_colors):
    idea_to_color =  {
        'day': my_colors[1], 
        "night": my_colors[2],
        "water ads": my_colors[0],
        "Riley": theme_colors[0],
        "Stovepipe": theme_colors[1],
        "Socorro": theme_colors[5],
        "Utqiagvik": theme_colors[3],
        "fitness": theme_colors[4],
        "step": theme_colors[5]
    }
    idea_to_color["ads"] = idea_to_color["night"]
    idea_to_color["des"] = idea_to_color["day"]
    return (idea_to_color,)


@app.cell
def _(os):
    fig_dir = "figs"
    os.makedirs(fig_dir, exist_ok=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 🌧️ modeling the vapor pressure of water
    """)
    return


@app.function
# input  T  : deg C
# output P* : bar
def water_p0(T):
    # coefficients for the following setup:
    #  log10(P) = A − (B / (T + C))
    #     P = vapor pressure (bar)
    #     T = temperature (K)
    # coefs from NIST https://webbook.nist.gov/cgi/cbook.cgi?ID=C7732185&Mask=4	
    # T in [293, 343] K
    if T+273.15 > 293.0 and T+273.15 < 343.0:
        A = 6.20963
        B = 2354.731
        C = 7.559
    # T in [273., 303] K
    elif T+273.15 > 273.0 and T+273.15 > 303.0:
        A = 5.40221
        B = 1838.675
        C = -31.737
    # T in [255.9, 373.] K
    elif (T+273.15 > 255.9 and T+273.15 < 373.0) or T < 255.9: # low temp
        A = 4.6543
        B = 1435.264
        C = -64.848
    # T in [379, 573] K
    elif T+273.15 > 379.0 and T+273.15 < 573.0: # high temp
        A = 3.55959
        B = 643.748
        C = -198.043
    else:
        raise Exception(f"T {T} not covered!")

    return 10.0 ** (A - B / ((T + 273.15) + C))


@app.cell
def _():
    water_p0(100.0) # around 1 ATM
    return


@app.cell
def _():
    water_p0(20.0) # 0.023 atm
    return


@app.cell
def _(np, plt):
    def viz_water_p0():
        Ts = np.linspace(-5.0, 100.0, 250) # deg C
        plt.figure()
        plt.xlabel("T [°C]")
        plt.ylabel("P* [bar]")
        plt.plot(Ts, [water_p0(T_i) for T_i in Ts], linewidth=3)
        plt.scatter(100.0, water_p0(100.0))
        plt.show()

    viz_water_p0()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # ☀️ weather time series data

    NOAA hourly data [here](https://www.ncei.noaa.gov/access/crn/products.html).

    (download directly; place in `data` subfolder.)
    """)
    return


@app.cell
def _(np):
    # temperature range
    T_range = [-20.0, 70.0] # deg C

    # ticks for plots
    T_ticks = np.linspace(T_range[0], T_range[1], 7)
    p_ovr_p0_ticks = np.linspace(0, 1, 6)
    return T_range, T_ticks, p_ovr_p0_ticks


@app.cell
def _(T_range, np):
    np.linspace(T_range[0], T_range[1], 7)
    return


@app.cell
def _():
    city_to_state = {
        'Tucson': 'AZ', 
        'Socorro': 'NM', 
        'Utqiagvik': 'AK', 
        'Mercury': 'NV', # Mojave desert
        'Stovepipe': 'CA',
        'Riley': 'OR', # high desert
        'Yuma': 'AZ' # Sonoran desert
    }
    return


@app.cell
def _():
    city_to_coords = {
        'Tucson':    (-110.9742, 32.2540),
        'Socorro':   (-106.8914, 34.0584), 
        'Mercury':   (-115.9945, 36.6605), 
        'Stovepipe': (-117.1465, 36.6062),
        'Riley':     (-119.5038, 43.5415),
        'Yuma':      (-114.6277, 32.6927),
        "Utqiagvik": (-156.788605, 71.290558)
    }
    return (city_to_coords,)


@app.cell
def _(ccrs, cfeature, city_to_coords, idea_to_color, plt):
    def viz_cities(cities):
        fig, ax = plt.subplots(
            figsize=(4, 4),
            subplot_kw={"projection": ccrs.PlateCarree()}
        )

        # Add map features
        ax.add_feature(cfeature.LAND)
        ax.add_feature(cfeature.OCEAN)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5)
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.STATES, linewidth=0.5, edgecolor="gray")
        ax.set_extent([-168, -100, 30, 73])  # USA bounds

        for city in cities:
            lon = city_to_coords[city][0]
            lat = city_to_coords[city][1]
            ax.plot(lon, lat, marker="*", markersize=15, color=idea_to_color[city],
                    transform=ccrs.PlateCarree())
            ax.text(lon, lat + 1.5, city, fontsize=10, ha="center",
                    transform=ccrs.PlateCarree())
        savename = "figs/map"
        for city in cities:
            savename = savename + "_" + city
        plt.tight_layout()
        plt.savefig(savename + ".pdf", format="pdf")
        plt.show()
    return (viz_cities,)


@app.cell
def _(mixed_locations, viz_cities):
    viz_cities(mixed_locations)
    return


@app.cell
def _(T_range, idea_to_color, np, os, pd, plt):
    class WeatherData:
        """
        read in weather time series from a location in a given month and year
        """
        def __init__(
            self, location, months, year, 
            verbose=True, time_to_hour={'day': 15, 'night': 5}
        ):
            self.location = location
            self.months = months
            self.year = year
            if verbose:
                print(f"loc: {location}. months: {months}/{year}")
                print("\tnighttime adsorption hr: ", time_to_hour["night"])
                print("\tdaytime harvest hr: ",      time_to_hour["day"])

            self.relevant_weather_cols = ["T_HR_AVG", "RH_HR_AVG", "SUR_TEMP", "SUR_RH_HR_AVG"]

            self.verbose = verbose
            self.raw_data = None
            self.ads_des_conditions = None
            self.all_missing = False

            self._read_raw_data()
            self._filter_missing()
            self._process_datetimes()
            self._infer_surface_RH()
            self._prune_raw_data()

            self.time_to_hour = time_to_hour
            self._attach_ads_des_conditions()

        def _read_raw_data(self):
            # search for unique relevant file
            wdata_dir = "data/"
            wfiles = os.listdir(wdata_dir)

            filename = list(
                filter(
                    lambda wf: self.location in wf and str(self.year) in wf, 
                    wfiles
                )
            )
            assert len(filename) == 1
            filename = wdata_dir + "/" + filename[0]
            if self.verbose:
                print(f"\treading raw data from {filename}")

            # column names
            names = open(wdata_dir + "/headers.txt", "r").readlines()[1].split()

            self.raw_data = pd.read_csv(
                filename,
                names=names, 
                dtype={'LST_DATE': str}, 
                sep='\s+'
            )

        def _process_datetimes(self):
            # convert to pandas datetime
            self.raw_data["date"] = pd.to_datetime(self.raw_data["LST_DATE"])

            # keep only the desired year
            self.raw_data = self.raw_data[
                self.raw_data["date"].dt.year == self.year
            ]

            # get hours
            self.raw_data["time"] = [
                pd.Timedelta(hours=h) for h in self.raw_data["LST_TIME"] / 100
            ]

            self.raw_data["datetime"] = (
                self.raw_data["date"] + self.raw_data["time"]
            )

            # filter by month
            self.raw_data = self.raw_data.loc[
                self.raw_data["datetime"].dt.month.isin(self.months)
            ]

            if self.raw_data.shape[0] == 0:
                print("\tWARNING: no data avail!")
                self.all_missing = True

        def _filter_missing(self):
            ids_bad = self.raw_data["T_HR_AVG"] < -999.0
            n_bad = np.sum(ids_bad)
            if n_bad > 0:
                print(f"\tfiltering {n_bad} missing rows in raw data")
                self.raw_data = self.raw_data[~ ids_bad]

        def _prune_raw_data(self):
            if self.all_missing:
                return
            self.raw_data = self.raw_data[
                ["datetime"] + self.relevant_weather_cols
            ]

        def _infer_surface_RH(self):
            # compute new relative humidity at surface temperature, 
            #     for heated air
            # partial pressure @ ambient:
            #      RH * p0(T)
            #         =
            # partial pressure @ surface:
            #   SUR_RH * p0(SUR_T)
            # => SUR_RH = RH * p0(T) / p0(SUR_T)
            if self.all_missing:
                return

            self.raw_data["SUR_RH_HR_AVG"] = self.raw_data.apply(
                lambda day: day["RH_HR_AVG"] * water_p0(day["T_HR_AVG"])
                    / water_p0(day["SUR_TEMP"]), 
                axis=1
            )

        def _attach_ads_des_conditions(self):
            cols_to_put = [
                'date', 'ads T [°C]', 'ads P/P0', 
                'des T [°C]', 'des P/P0'#, 'rained'
            ]

            if self.all_missing:
                self.ads_des_conditions = pd.DataFrame(columns=cols_to_put + ['location'])
                return
            # get separate day and night data frames with precise time stamp
            # useful for checking and for plotting as 
            #    a time series with all of the data
            wdata = dict()
            for time in ["day", "night"]:
                wdata[time] = self.raw_data[
                    self.raw_data["datetime"].dt.hour == self.time_to_hour[time]
                ].rename(
                    columns={
                        col: time + "_" + col 
                        for col in self.relevant_weather_cols
                    }
                )
                wdata[time]["date"] = wdata[time]["datetime"].dt.normalize()

            self.ads_des_conditions = pd.merge(
                wdata["night"], wdata["day"],
                on="date", how="inner"
            )

            self.ads_des_conditions.sort_values(by="date", inplace=True)

            self.ads_des_conditions = self.ads_des_conditions.rename(
                columns=
                {
                    # adsorptin conditions (night)
                    "night_T_HR_AVG": 'ads T [°C]',
                    "night_RH_HR_AVG": 'ads P/P0',
                    # desorption conditions (day)
                    "day_SUR_TEMP": 'des T [°C]',
                    "day_SUR_RH_HR_AVG": 'des P/P0',
                    # # rain col
                    # "day_rain_daily_total": "rain_daily_total"
                }
            )
            for rh_col in ['des P/P0', 'ads P/P0']:
                self.ads_des_conditions[rh_col] = (
                    self.ads_des_conditions[rh_col] / 100.0
                )

            self.ads_des_conditions = self.ads_des_conditions[cols_to_put]

            self.ads_des_conditions["location"] = self.location

            # for a reasonable average... missing data otherwise.
            assert self.ads_des_conditions["date"].nunique() > 25

        def viz_timeseries(
            self, save=False, incl_legend=True, 
            legend_dx=0.0, legend_dy=0.0
        ):
            ads = {'air': "k", 'surface': "k"}

            fig, axs = plt.subplots(2, 1, sharex=True)
            plt.xticks(rotation=90, ha='center')

            ###
            #   temperature
            ###
            axs[0].set_ylabel("temperature\n[°C]")
            for ads_des in ["ads", "des"]:
                axs[0].scatter(
                    self.ads_des_conditions["date"], 
                    self.ads_des_conditions[f"{ads_des} T [°C]"],
                    edgecolors="black", clip_on=False,
                    marker="^", 
                    color=idea_to_color[ads_des], zorder=10, 
                    label=ads_des, 
                    s=25
                )

            ###
            #   relative humidity
            ###
            axs[1].set_ylabel("relative\nhumidity")
            for ads_des in ["ads", "des"]:
                axs[1].scatter(
                    self.ads_des_conditions["date"], 
                    self.ads_des_conditions[f"{ads_des} P/P0"],
                    edgecolors="black", clip_on=False,
                    marker="v", 
                    color=idea_to_color[ads_des], zorder=10, 
                    label=ads_des, 
                    s=25
                )
            axs[1].legend(
                    prop={'size': 10}, ncol=1, 
                    bbox_to_anchor=(0., 1.0 + legend_dy, 1.0 + legend_dx, .1),
                   loc="center left"
            )#, loc="center left")

            plt.show()

        def _assert_in_T_range(self):
            T_min = self.ads_des_conditions[
                ["ads T [°C]", "des T [°C]"]
            ].min().min()
            T_max = self.ads_des_conditions[
                ["ads T [°C]", "des T [°C]"]
            ].max().max()

            # manually set
            if T_min < T_range[0] or T_max > T_range[1]:
                print([T_min, T_max])
                raise Exception("extend T_range")

        def sees_below_zeroC(self):
            return self.ads_des_conditions[
                ["ads T [°C]", "des T [°C]"]
            ].min().min() < 0.0
    return (WeatherData,)


@app.cell
def _(WeatherData):
    wdata = WeatherData("Stovepipe", [5], 2021)
    wdata.ads_des_conditions["date"]
    return (wdata,)


@app.cell
def _(wdata):
    wdata.viz_timeseries()
    return


@app.cell
def _(T_range, pd):
    class Weather:
        """
        combine weather time series data from multiple (location, month, year)
          combinations
        """
        def __init__(
            self, weather_datas, tag
        ):
            self.tag = tag

            self.ads_des_conditions = (
                pd.concat(
                    [w.ads_des_conditions for w in weather_datas], 
                    ignore_index=True
                ).sort_values("date").reset_index(drop=True)
            )

            n_rows = self.ads_des_conditions.shape[0]

            self._assert_in_T_range()
            print("# days:", self.ads_des_conditions.shape[0])

        def _assert_in_T_range(self):
            T_min = self.ads_des_conditions[
                ["ads T [°C]", "des T [°C]"]
            ].min().min()
            T_max = self.ads_des_conditions[
                ["ads T [°C]", "des T [°C]"]
            ].max().max()

            # manually set
            if T_min < T_range[0] or T_max > T_range[1]:
                print([T_min, T_max])
                raise Exception("extend T_range")
    return (Weather,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## choose location
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    dropdown = mo.ui.dropdown(
        options=["Yuma", "Riley", "Stovepipe", "Mercury", "Socorro", "Utqiagvik", "mix"], 
        value="mix", label="choose location"
    )
    # Stovepipe: opt at 0.072
    # Socorro: opt at 0.15
    # Riely: opt at 0.312
    # Utqiagvik: opt at 0.8
    dropdown
    return (dropdown,)


@app.cell
def _():
    too_many_missing = [
        ["Stovepipe", 5, 2021],
        ["Utqiagvik", 5, 2021],
        ["Socorro", 6, 2022]
    ]
    return (too_many_missing,)


@app.cell
def _(WeatherData, np, too_many_missing):
    # Socorro: opt step at 0.154
    def get_weather_datas(locations, months, years, randomize_location=True):
        weather_datas = []
        for yr in years:
            for mo in months:
                locs_to_use = [np.random.choice(locations)] if randomize_location else locations
                for location in locs_to_use:
                    if [location, mo, yr] in too_many_missing:
                        print("SKIPPING: too much missing.")
                        continue
                    wdata = WeatherData(location, [mo], yr)
                    weather_datas.append(wdata)
        return weather_datas
    return (get_weather_datas,)


@app.cell
def _(Weather, dropdown, get_weather_datas):
    summer_months = [6, 7, 8] # meterological
    summer_months = [5, 6, 7, 8, 9]
    yrs = [2019, 2020, 2021, 2022, 2023, 2024, 2025]

    mixed_locations = ["Stovepipe", "Socorro", "Riley", "Utqiagvik"]

    weather_datas = []
    if not dropdown.value == "mix":
        weather_datas = get_weather_datas([dropdown.value], summer_months, yrs)
    elif dropdown.value == "mix":
        weather_datas = get_weather_datas(mixed_locations, summer_months, yrs, randomize_location=False)

    weather = Weather(
        # list of weather data
        weather_datas,
        # tag
        dropdown.value
    )
    weather.ads_des_conditions
    return mixed_locations, weather


@app.cell
def _(weather):
    for wmetric in ["ads T [°C]", "des T [°C]", "ads P/P0", "des P/P0"]:
        print(wmetric)
        print("\tmin = ", weather.ads_des_conditions[wmetric].min())
        print("\tmax = ", weather.ads_des_conditions[wmetric].max())
        print("\tmean = ", weather.ads_des_conditions[wmetric].mean())
        print("\tstd = ", weather.ads_des_conditions[wmetric].std())

    print(
        "mean delta p/p0: ", (
            weather.ads_des_conditions["ads P/P0"] - \
            weather.ads_des_conditions["des P/P0"]
        ).mean()
    )
    return


@app.cell
def _(
    T_range,
    T_ticks,
    dropdown,
    idea_to_color,
    mixed_locations,
    p_ovr_p0_ticks,
    plt,
    sns,
    weather,
):
    with sns.plotting_context("notebook", font_scale=1.4):
        short_to_proper_weather_cols = {
            'ads T [°C]': 'capture $T$ [°C]',
            'des T [°C]': 'release $T$ [°C]',
            'ads P/P0': 'capture $p/p_0(T)$',
            'des P/P0': 'release $p/p_0(T)$',
        }

        weather_cols = ['ads P/P0', 'ads T [°C]', 'des P/P0', 'des T [°C]']

        pp = sns.pairplot(
            weather.ads_des_conditions.rename(
                columns=short_to_proper_weather_cols
            ),
            vars=[short_to_proper_weather_cols[w] for w in weather_cols],
            kind="kde",
            hue="location",
            hue_order=mixed_locations if dropdown.value == "mix" else [dropdown.value],
            palette=idea_to_color,
            corner=True,
            plot_kws=dict(linewidths=1, fill=True, alpha=0.6),
            diag_kws=dict(linewidths=1, fill=True, alpha=0.6),
            diag_kind='kde'
        )

        for ax in [pp.axes[2, 0], pp.axes[3, 1]]:
            ax.axline(
                (0, 0), (1, 1), transform=ax.transAxes, color='black', 
                zorder=0, linestyle="--"
            )

        def set_weather_cols_axis(pp):
            for c in range(4):
                pp.axes[-1, c].tick_params(axis='x', labelrotation=90)
            for r in [1, 3]:
                pp.axes[r, 0].set_ylim(T_range)
                pp.axes[r, 0].set_yticks(T_ticks)

            pp.axes[2, 0].set_ylim(0, 1.0)
            pp.axes[2, 0].set_yticks(p_ovr_p0_ticks)
            for c in [0, 2]:
                pp.axes[3, c].set_xlim(0, 1)
                pp.axes[3, c].set_xticks(p_ovr_p0_ticks)
            for c in [1, 3]:
                pp.axes[3, c].set_xlim(T_range)
                pp.axes[3, c].set_xticks(T_ticks)

        set_weather_cols_axis(pp)

        # plt.savefig(
        #     weather.tag + "ads_des_conditions.pdf", 
        #     format="pdf"
        # )
        plt.show()
    return set_weather_cols_axis, short_to_proper_weather_cols, weather_cols


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 🛏️ modeling a water adsorption isotherm in a MOF bed

    ## Bernstein polynomial composition
    """)
    return


@app.cell
def _(comb, np):
    class BernPolyBasis:
        def __init__(self, n):
            self.n = n
            self.vs = np.arange(n + 1)
            self.coeffs = comb(n, self.vs) # pre-compute for speed

        def basis_matrix(self, x):
            """
            Evaluate all n+1 basis functions at every point in x.
            x: scalar or array-like, shape (m,)
            returns: array, shape (m, n+1)
            """
            x = np.atleast_1d(np.asarray(x, dtype=float))
            return (
                self.coeffs
                * x[:, None] ** self.vs[None, :]
                * (1.0 - x[:, None]) ** (self.n - self.vs[None, :])
            )

        def value(self, x, bs):
            """
            x: scalar or array-like, shape (m,)
            bs: array-like, shape (n+1,) — control points
            returns: scalar (if x was scalar) or array, shape (m,)
            """
            x = np.asarray(x, dtype=float)
            assert np.all((x >= 0.0) | np.isnan(x))
            assert np.all((x <= 1.0) | np.isnan(x))
            scalar_input = (x.ndim == 0)
            basis = self.basis_matrix(x)       # (m, n+1)
            val = basis @ np.asarray(bs)       # (m,)
            return val[0] if scalar_input else val
    return (BernPolyBasis,)


@app.cell
def _(BernPolyBasis, np, plt):
    def viz_bern(n):
        xs = np.linspace(0.0, 1.0, 250)

        bp = BernPolyBasis(n)
        basis_matrix = bp.basis_matrix(xs)

        fig = plt.figure()
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.xlabel(r"$\phi_0 := P/P_0$")
        plt.ylabel(r"$b_{\nu, n}(\phi_0)$")
        for v in range(n+1):
            plt.plot(xs, basis_matrix[:, v], label=rf"$\nu={v}$")
        plt.legend()
        plt.title(rf"$n={n}$")
        plt.tight_layout()
        plt.savefig("bernstein_basis_polys.pdf", format="pdf")
        plt.show()

    viz_bern(4)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## water adsorption isotherm class
    """)
    return


@app.cell
def _(BernPolyBasis, colors, inset_axes, mpl, np, plt):
    class WaterAdsorptionIsotherm:
        def __init__(
            self, n, Tref=25.0, w_max=0.5, bs=None
        ):
            # number of control points
            self.n = n
            self.bp = BernPolyBasis(n)

            # max water ads [kg H2O/kg MOF]
            self.w_max = w_max

            # reference temperature [deg. C]
            self.Tref = Tref

            self.label = None

            # pre-allocate bs
            if bs is None:
                self.bs = np.full(n + 1, np.nan)
            else:
                self.bs = bs

        def copy(self):
            return WaterAdsorptionIsotherm(
                self.n, Tref=self.Tref,
                w_max=self.w_max, bs=np.copy(self.bs)
            )

        def endow_random_isotherm(self):
            self.bs[1:-1] = np.sort(np.random.rand(self.n - 1)) * self.w_max
            self.bs[0]  = 0.0 # start at zero
            self.bs[-1] = self.w_max # end at 1

        def endow_stepped_isotherm(self, i):
            self.bs[:i] = 0.0
            self.bs[i:] = self.w_max

        def endow_random_stepped_isotherm(self):
            i = np.random.choice(self.n+1)
            self.endow_stepped_isotherm(i)    

        def water_ads(self, T, p_over_p0):
            """
            water adsorption in this MOF
            - T: deg C
            - p/p0(T) : unitless
            """
            # model: expand adsorption n as a function of 
            #   phi_ref = p / p0[T_ref]
            #   with Bernstein polynomial basis functions.
            # Polanyi: A = - R T log(p / p0[T])
            #          n = n(A)
            # set A = - RT log(phi) = - R T_ref log(phi_ref)
            #   cuz we wanna know corresponding phi_ref at
            #   T_ref that gives same A at T
            #   so T / T_Ref log(phi) = log(phi_ref)
            #      log(phi^(T/T_Ref)) = log(phi_ref) 
            p_over_p0_ref = p_over_p0 ** ((T + 273.15) /  (self.Tref + 273.15))

            return self.bp.value(p_over_p0_ref, self.bs)

        def water_del(self, conditions):
            w_ads = self.water_ads(
                conditions["ads T [°C]"].to_numpy(),
                conditions["ads P/P0"].to_numpy(),
            )
            w_des = self.water_ads(
                conditions["des T [°C]"].to_numpy(),
                conditions["des P/P0"].to_numpy(),
            )
            return np.where(w_ads > w_des, w_ads - w_des, 0.0)

        def water_del_distn(self, weather):
            w_dels = self.water_del(weather.ads_des_conditions)

            plt.figure()
            plt.hist(w_dels)
            plt.ylabel("# days")
            plt.xlabel("water delivery")
            plt.show()

        def get_p_ovr_p0_half_max(self, verbose=False):
            p_over_p0s = np.linspace(0, 1.0, 500)
            ws = np.array(
                [
                    self.water_ads(self.Tref, p_over_p0)
                    for p_over_p0 in p_over_p0s
                ]
            )
            id = np.argmax(ws > self.w_max / 2)
            p_star = p_over_p0s[id]
            if verbose:
                print(f"ads at T={self.Tref}deg C half max at p/p0 = ", p_star)
            return p_star

        def draw(self):
            p_over_p0s = np.linspace(0, 1.0, 100)

            fig, ax = plt.subplots()

            plt.xlabel("relative humidity $p / [p_0(T)]$")
            plt.ylabel("water adsorption [kg H$_2$O/kg sorbent]")

            colormap = mpl.colormaps['coolwarm'] # or 'plasma', 'coolwarm', etc.
            norm = colors.Normalize(vmin=0.0, vmax=70.0)

            for T in np.linspace(0, 70, 6):
                plt.plot(
                    p_over_p0s, 
                    [self.water_ads(T, p_over_p0) for p_over_p0 in p_over_p0s],
                    color=colormap(norm(T)),
                    clip_on=False
                )

            sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
            cax = inset_axes(
                ax, width="4%", height="40%", loc="lower right",
                bbox_to_anchor=(0.0, 0.05, 0.9, 0.95),  # (x0, y0, width, height) in axes fraction
                bbox_transform=ax.transAxes, borderpad=0
            )
            fig.colorbar(sm, cax=cax, label='temperature [°C]')
            ax.set_xlim(0, 1.0)
            ax.set_ylim(0, self.w_max)

            plt.tight_layout()

            plt.show()
    return (WaterAdsorptionIsotherm,)


@app.cell
def _(WaterAdsorptionIsotherm):
    wai = WaterAdsorptionIsotherm(10)
    wai.endow_stepped_isotherm(3)
    wai.draw()
    return (wai,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 🥇 score fitness of dist'n of water deliveries
    """)
    return


@app.function
def attach_water_delivery(wai, weather):
    # compute water delivery
    weather.ads_des_conditions["water del [kg H$_2$O/kg MOF]"] = wai.water_del(
        weather.ads_des_conditions
    )


@app.function
def get_monthly_totals(wai, weather):
    attach_water_delivery(wai, weather)

    monthly_totals = (
        weather.ads_des_conditions
        .groupby(["location", weather.ads_des_conditions["date"].dt.to_period("M")])["water del [kg H$_2$O/kg MOF]"]
        .sum()
    )
    
    return monthly_totals


@app.cell
def _(wai, weather):
    get_monthly_totals(wai, weather)
    return


@app.cell
def _(np):
    def var_cvar(scores, alpha):
        val_at_risk = np.percentile(scores, alpha)
        cval_at_risk = np.mean(scores[scores <= val_at_risk])
        return val_at_risk, cval_at_risk
    return (var_cvar,)


@app.cell
def _():
    alpha = 10.0
    return (alpha,)


@app.cell
def _(alpha, var_cvar):
    def score_fitness(wai, weather, alpha=alpha):
        monthly_totals = get_monthly_totals(wai, weather)

        scores = monthly_totals.values

        val_at_risk, cval_at_risk = var_cvar(scores, alpha)

        return scores, cval_at_risk
    return (score_fitness,)


@app.cell
def _(score_fitness, wai, weather):
    scores, fitness = score_fitness(wai, weather)
    return


@app.cell
def _(matplotlib, np):
    def draw_fitness_ax(
        ax, scores, fitness, color, label, 
        alpha=10, orientation="vertical"
    ):
        max_score = 0.5 * 32
        bins = np.linspace(0, max_score, 17)
        assert np.max(scores) < max_score

        face_rgba = matplotlib.colors.to_rgba(color, alpha=0.5)
        edge_rgba = matplotlib.colors.to_rgba(color, alpha=1.0)

        ax.hist(
            scores, bins=bins, histtype="stepfilled",
            facecolor=face_rgba, edgecolor=edge_rgba, linewidth=1.5, label=label
        )

        ax.set_xlabel("total water delivered\n[kg H$_2$O/kg sorbent]")
        ax.set_ylabel("# months")   
        ax.set_xlim([0.0, max_score])
        ax.set_ylim(ymin=0.0)

        ax.axvline(fitness, linestyle="--", color=color)
    return (draw_fitness_ax,)


@app.cell
def _(draw_fitness_ax, idea_to_color, plt, score_fitness):
    def draw_fitness_scores(wai, weather):
        scores, fitness = score_fitness(wai, weather)
        print("fitness [kg/kg]: ", fitness)

        plt.figure()
        draw_fitness_ax(
            plt.gca(),
            scores, 
            fitness, 
            idea_to_color["fitness"],
            ""
        )
        plt.tight_layout()

        # plt.savefig(weather.tag + "eg_var.pdf", format="pdf")
        plt.show()
    return (draw_fitness_scores,)


@app.cell
def _(draw_fitness_scores, wai, weather):
    draw_fitness_scores(wai, weather)
    return


@app.cell
def _(
    alpha,
    calendar,
    dropdown,
    idea_to_color,
    plt,
    sns,
    var_cvar,
    wai,
    weather,
):
    def viz_monthly_water_del(wai, weather, legend_outside=dropdown.value == "mix"):
        monthly_totals = get_monthly_totals(wai, weather).reset_index()
        monthly_totals["month"] = monthly_totals["date"].dt.month
        fitness = var_cvar(monthly_totals["water del [kg H$_2$O/kg MOF]"].values, alpha)[1]

        # rename for seaborn
        monthly_totals = monthly_totals.rename(
            columns = {
                "water del [kg H$_2$O/kg MOF]": "total water delivered\n[kg H$_2$O/kg MOF]"
            }
        )
    
        sns.swarmplot(
            data=monthly_totals, 
            y="total water delivered\n[kg H$_2$O/kg MOF]", 
            x="month", 
            palette=idea_to_color,
            hue="location",
            size=10
        )

        mos = monthly_totals["month"].unique()
        plt.xticks(range(len(mos)), [calendar.month_name[mo] for mo in mos])

        plt.axhline(fitness, color=idea_to_color["fitness"], linestyle="--", zorder=0)
        plt.ylim([0, 16])
        if legend_outside:
            plt.legend(
                title="location",
                bbox_to_anchor=(1.02, 1),
                loc="upper left",
                borderaxespad=0
            )
        plt.show()

    viz_monthly_water_del(wai, weather)
    return (viz_monthly_water_del,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 🎲 random WAIs to explore
    """)
    return


@app.cell
def _(draw_fitness_ax, my_colors, np, p_ovr_p0_ticks, plt, score_fitness):
    def compare_wais(wais, weather, savetag=""):
        the_colors = [my_colors[0]] + my_colors[3:]
        p_over_p0s = np.linspace(0, 1.0, 100)

        fig = plt.figure(figsize=(4, 6), layout="constrained")
        gs = fig.add_gridspec(2, 1)
        ax_top = fig.add_subplot(gs[0, 0])
        ax_bot = fig.add_subplot(gs[1, 0])

        ###
        #   adsorption isotherm
        ###
        ax_bot.set_xlabel("relative humidity, $p / [p_0(T)]$")
        ax_bot.set_xticks(p_ovr_p0_ticks)
        ax_bot.set_ylabel(
            f"water adsorption at {wais[0].Tref:.0f}°C\n[kg H$_2$O/kg sorbent]"
        )

        for w, wai in enumerate(wais):
            ax_bot.plot(
                p_over_p0s, 
                [wai.water_ads(wai.Tref, p_over_p0) for p_over_p0 in p_over_p0s],
                color=the_colors[w],
                label=wai.label
            )

        ax_bot.set_xlim(0, 1)
        ax_bot.set_ylim(0, wais[0].w_max)
        ax_bot.legend(title="water ads. isotherm", fontsize=8, title_fontsize=10)

        ###
        #   fitness scores
        ###
        for w, wai in enumerate(wais):
            scores, fitness = score_fitness(wai, weather)
            print(f"fitness WAI {wai.label}: {fitness}")
            draw_fitness_ax(ax_top, scores, fitness, the_colors[w], label=w)

        plt.show()
    return (compare_wais,)


@app.cell
def _(WaterAdsorptionIsotherm, compare_wais, np, score_fitness, weather):
    _wais = [WaterAdsorptionIsotherm(10) for i in range(51)]
    for _wai in _wais:
        _wai.endow_random_isotherm()

    _fitness = [score_fitness(wai, weather)[1] for wai in _wais]
    _ids = np.argsort(_fitness)

    _wais = [
        _wais[_ids[0]], 
        _wais[np.where(_fitness == np.median(_fitness))[0][0]], 
        _wais[_ids[-1]]
    ]
    compare_wais(_wais, weather)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 🍃 evolutionary optimization
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 🦋 evolutionary operations
    """)
    return


@app.cell
def _(my_colors, np, p_ovr_p0_ticks, plt):
    def viz_wais(
        wais, savename=None, material_labels=None
    ):
        if material_labels is None:
            material_labels = [f"#{w}" for w in range(len(wais))]

        the_colors = [my_colors[0]] + my_colors[3:]
        p_over_p0s = np.linspace(0, 1.0, 100)

        fig = plt.figure(figsize=(4.5, 4))
        plt.xlabel("relative humidity $p / [p_0(T)]$")
        plt.xticks(p_ovr_p0_ticks)
        plt.ylabel(
            f"water adsorption at {wais[0].Tref:.0f}°C\n[kg H$_2$O/kg sorbent]"
        )

        for w, wai in enumerate(wais):
            plt.plot(
                p_over_p0s, 
                [wai.water_ads(wai.Tref, p_over_p0) for p_over_p0 in p_over_p0s],
                color=the_colors[w],
                label=material_labels[w]
            )

        plt.xlim(0, 1.0)
        plt.ylim(0, wais[0].w_max)
        plt.legend(title="model material", fontsize=8, title_fontsize=10)
        if savename is not None:
            plt.savefig(
                savename + ".pdf", format="pdf",  bbox_inches="tight"
            )
        plt.show()
    return (viz_wais,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    random birth        wai = WaterAdsorptionIsotherm(dim)
            if np.random.rand() < 0.5:
                wai.endow_random_stepped_isotherm()
            else:
                wai.endow_random_isotherm()
            new_wais.append(wai)
    """)
    return


@app.cell
def _(WaterAdsorptionIsotherm, np):
    def random_birth(n):
        wai = WaterAdsorptionIsotherm(n)
        if np.random.rand() < 0.5:
            wai.endow_random_stepped_isotherm()
        else:
            wai.endow_random_isotherm()
        return wai
    return (random_birth,)


@app.cell
def _(n, random_birth, viz_wais):
    viz_wais(
        [random_birth(n) for i in range(5)], savename="random_births"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    mutation
    """)
    return


@app.cell
def _(np):
    def mutate(wai, eps):
        # perturb
        delta_b = 2 * eps * (np.random.rand(wai.n - 1) - 0.5)

        # enforce constraint
        if np.random.rand() < 0.0:
            wai.bs[1:-1] += delta_b
            wai.bs = np.sort(wai.bs)
        else:
            wai.bs[1:-1] += np.sort(delta_b)

        wai.bs[wai.bs < 0.0] = 0.0
        wai.bs[wai.bs > wai.w_max] = wai.w_max
        wai.bs[-1] = wai.w_max
    return (mutate,)


@app.cell
def _(WaterAdsorptionIsotherm, mutate, viz_wais):
    _wais = [WaterAdsorptionIsotherm(20)]
    _wais[0].endow_random_isotherm()
    _wais.append(_wais[0].copy())
    mutate(_wais[1], 0.05)
    viz_wais(_wais, material_labels=["original", "mutated"], savename="mutation")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    tournament selection
    """)
    return


@app.cell
def _(np):
    def run_tournament(fitnesses, tourney_size):
        ids_tourney = np.random.choice(
            np.size(fitnesses), size=tourney_size, replace=False
        )

        # compete for top two (= the chosen parents)
        ids_winners = np.argpartition(fitnesses[ids_tourney], -2)[-2:]
        id_a = ids_tourney[ids_winners[0]]
        id_b = ids_tourney[ids_winners[1]]
        return id_a, id_b
    return (run_tournament,)


@app.cell
def _(np, run_tournament):
    run_tournament(np.arange(10), 5)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    combination
    """)
    return


@app.cell
def _(WaterAdsorptionIsotherm, np):
    def random_combination(wai_a, wai_b):
        alpha = np.random.rand() # fraction of genes of parent a

        return WaterAdsorptionIsotherm(
            wai_a.n, bs=alpha * wai_a.bs + (1 - alpha) * wai_b.bs
        )
    return (random_combination,)


@app.cell
def _(WaterAdsorptionIsotherm, random_combination, viz_wais):
    _rand_wais = [WaterAdsorptionIsotherm(20), WaterAdsorptionIsotherm(20)]
    _rand_wais[0].endow_stepped_isotherm(7)
    _rand_wais[1].endow_stepped_isotherm(15)
    _rand_wais.append(random_combination(_rand_wais[0], _rand_wais[1]))
    viz_wais(
        _rand_wais, 
        material_labels=["parent A", "parent B", "child"],
        savename="combination"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    cross-over
    """)
    return


@app.cell
def _(np):
    def random_cross_over(wai_a, wai_b, random_switch=True):
        # change up which gives left and right portion of isotherm
        if random_switch:
            if np.random.rand() < 0.5:
                return random_cross_over(wai_b, wai_a, random_switch=False)

        # swap point
        id = np.random.choice(range(wai_a.n))

        wai = wai_a.copy()               # wai_a gives left side
        wai.bs[id:] = wai_b.bs[id:]  # wai_b gives right side

        # enforce monotonicity
        wai.bs = np.sort(wai.bs)

        return wai
    return (random_cross_over,)


@app.cell
def _(WaterAdsorptionIsotherm, random_cross_over, viz_wais):
    _rand_wais = [WaterAdsorptionIsotherm(20), WaterAdsorptionIsotherm(20)]
    _rand_wais[0].endow_stepped_isotherm(5)
    _rand_wais[1].endow_random_isotherm()
    print("parent A:", _rand_wais[0].bs)
    print("parent B:", _rand_wais[1].bs)

    _rand_wais.append(random_cross_over(_rand_wais[0], _rand_wais[1]))
    print("child:", _rand_wais[-1].bs)
    viz_wais(
        _rand_wais, 
        material_labels=["parent A", "parent B", "child"],
        savename="crossover"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    local search through stepify
    """)
    return


@app.cell
def _(score_fitness):
    # increase capacity at high pressure until fitness decreases
    # decrease capacity at low pressure until fitness decreases
    def ls_stepify(wai, weather, verbose=False): 
        new_wai = wai.copy()

        scores, fitness = score_fitness(wai, weather)
        if verbose:
            print("---local search---")
            print("current fitness: ", fitness)

        # max out capacity at high p/p0 until fitness decreases
        for i in range(1, wai.n): # walk backwards thru array
            new_wai.bs[-i:] = wai.w_max
            new_scores, new_fitness = score_fitness(new_wai, weather)
            if verbose:
                print("new fitness: ", new_fitness)

            if new_fitness >= fitness: # OR EQUAL TO (important)
                if verbose:
                    print(
                        "maxed out uptake at high p/p0 w./o decrease in fitness."
                    )
                    print("\tnew fitness: ", new_fitness)
                wai.bs[:] = new_wai.bs
                fitness = new_fitness
            else:
                break 

        # destroy capacity at low p/p0 until fitness decreases
        for i in range(1, wai.n): # walk forwards thru array
            new_wai.bs[:i] = 0.0
            new_scores, new_fitness = score_fitness(new_wai, weather)
            if verbose:
                print("new fitness: ", new_fitness)

            if new_fitness >= fitness: # OR EQUAL TO (important)
                if verbose:
                    print(
                        "zeroed uptake at low p/p0 w./o decrease in fitness."
                    )
                    print("\tnew fitness: ", new_fitness)
                wai.bs[:] = new_wai.bs
                fitness = new_fitness
            else:
                break 
    return (ls_stepify,)


@app.cell
def _(WaterAdsorptionIsotherm, ls_stepify, viz_wais, weather):
    _wai = WaterAdsorptionIsotherm(20)
    _wai.endow_random_isotherm()
    _wai2 = _wai.copy()
    ls_stepify(_wai2, weather, verbose=False)
    viz_wais(
        [_wai, _wai2], 
        material_labels=["original", "stepified"],
        savename="stepify"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## evolution step
    """)
    return


@app.cell
def _(
    ls_stepify,
    mutate,
    np,
    random_birth,
    random_combination,
    random_cross_over,
    run_tournament,
    score_fitness,
    warnings,
):
    def evolve(
        wais, weather, n_elite=5, tourney_size=10, 
        n_rand=15, n_mutate=15, eps=0.05, verbose=False
    ):
        # what's the population size?
        pop_size = np.shape(wais)[0]

        # max water adsorption
        w_max = wais[0].w_max

        # dimension of search space
        dim = wais[0].n

        # compute fitnesses of each individual
        fitnesses = np.array([score_fitness(wai, weather)[1] for wai in wais])

        # which are the elite individuals?
        ids_elite = np.argpartition(fitnesses, -n_elite)[-n_elite:]

        if np.all(fitnesses[ids_elite[0]] == fitnesses[ids_elite]):
            warnings.warn("elite class all same fitness!")

        if verbose:
            print("initial generation")
            print("\telite fitness: ", fitnesses[ids_elite])

        # initiate new generation with the elite individuals un-modified
        new_wais = [wais[i_elite] for i_elite in ids_elite]
        # local search
        for elite_wai in new_wais:
            if np.random.rand() < 0.2:
                ls_stepify(elite_wai, weather)

        # tournament selection
        for i in range(pop_size - n_elite - n_rand):
            id_a, id_b = run_tournament(fitnesses, tourney_size)

            # mate to produce child
            if np.random.rand() < 0.5:
                new_wai = random_cross_over(wais[id_a], wais[id_b])
            else:
                new_wai = random_combination(wais[id_a], wais[id_b])

            new_wais.append(new_wai)

        # random births for exploration
        for i in range(n_rand):
            new_wais.append(
                random_birth(dim)
            )

        # mutation
        for i in range(n_mutate):
            # select non-elite individual to mutate
            id = np.random.choice(np.arange(n_elite, pop_size))
            mutate(new_wais[id], eps)

        return new_wais
    return (evolve,)


@app.cell
def _(random_birth):
    def gen_initial_pop(pop_size, n):
        return [random_birth(n) for _ in range(pop_size)]
    return (gen_initial_pop,)


@app.cell
def _(evolve, gen_initial_pop, np, plt, score_fitness, weather):
    # first generation
    wais = gen_initial_pop(75, 25)

    fitnesses = np.array([score_fitness(wai, weather)[1] for wai in wais])

    # second generation
    new_wais = evolve(wais, weather, n_elite=5)
    new_fitnesses = np.array(
        [score_fitness(new_wai, weather)[1] for new_wai in new_wais]
    )

    plt.figure()
    plt.xlabel("fitness")
    plt.ylabel("# soln's")
    plt.hist(fitnesses, alpha=0.5, label="gen #0")
    plt.hist(new_fitnesses, alpha=0.5, label="gen #1")
    plt.legend()
    plt.show()
    return (wais,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 👟 run the evol algo
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    run_evol_cbox = mo.ui.checkbox(label="run evolution algo")
    run_evol_cbox
    return (run_evol_cbox,)


@app.cell
def _(evolve, gen_initial_pop, np, score_fitness):
    def do_evolution(weather, n_generations, pop_size, dim):
        # generate population
        wais = gen_initial_pop(pop_size, dim)

        # score fitnesses
        fitnesses = np.array([score_fitness(wai, weather)[1] for wai in wais])

        # store progress
        fitnesses_gen = [fitnesses]
        best_wai_gen = [wais[np.argmax(fitnesses)]]

        # evolve over generations
        for g in range(1, n_generations):
            wais = evolve(wais, weather)
            fitnesses = np.array([score_fitness(wai, weather)[1] for wai in wais])

            fitnesses_gen.append(fitnesses)
            best_wai_gen.append(wais[np.argmax(fitnesses)])

        best_wai = wais[np.argmax(fitnesses)]
        best_wai.label = "optimal"
        best_scores, best_fitness = score_fitness(best_wai, weather)

        return fitnesses_gen, best_wai_gen, best_wai, best_scores, best_fitness
    return (do_evolution,)


@app.cell
def _(do_evolution, run_evol_cbox, weather):
    pop_size = 30
    n_generations = 25
    n = 50
    if run_evol_cbox.value:
        fitnesses_gen, best_wai_gen, best_wai, best_scores, best_fitness = do_evolution(
            weather, n_generations, pop_size, n
        )
    return best_fitness, best_wai, best_wai_gen, fitnesses_gen, n


@app.cell
def _(best_wai):
    best_wai.draw()
    return


@app.cell
def _(best_wai):
    best_wai.get_p_ovr_p0_half_max()
    return


@app.cell
def _():
    return


@app.cell
def _(best_wai, draw_fitness_scores, weather):
    draw_fitness_scores(best_wai, weather)
    return


@app.cell
def _(best_wai, weather):
    monthly_water_del = get_monthly_totals(best_wai, weather)
    monthly_water_del
    return (monthly_water_del,)


@app.cell
def _(monthly_water_del):
    monthly_water_del.reset_index().sort_values("water del [kg H$_2$O/kg MOF]")
    return


@app.cell
def _(best_wai, viz_monthly_water_del, weather):
    viz_monthly_water_del(best_wai, weather)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### analyze progress
    """)
    return


@app.cell
def _(fitnesses_gen, pd, plt, sns):
    def viz_fitness_progress(fitnesses_gen):
        data = pd.DataFrame(
            [
                [g, fitness] for g, fitnesses in enumerate(fitnesses_gen) 
                for fitness in fitnesses
            ]
            ,
            columns=['generation', 'fitness [kg H$_2$O/kg sorbent]']
        )

        fig, ax = plt.subplots(figsize=(6, 4))

        sns.stripplot(
            data, 
            x="generation", y="fitness [kg H$_2$O/kg sorbent]",
            hue="generation", color="C2", palette="crest", legend=False,
            ax=ax, clip_on=False
        )
        plt.tick_params(axis='x', labelrotation=90)
        # plt.axhline(
        #     y=step_fitnesses[id_opt_step], 
        #     color="gray", linestyle="--", zorder=-1
        # )
        plt.ylim(ymin=0)
        plt.tight_layout()
        # plt.savefig(
        #     weather.tag + "fitness_progress.pdf", format="pdf"
        # )
        plt.show()

    viz_fitness_progress(fitnesses_gen)
    return


@app.cell
def _(best_wai_gen, colors, mpl, np, p_ovr_p0_ticks, plt, wais, weather):
    def viz_best_wais(best_wai_gen):
        p_over_p0s = np.linspace(0, 1.0, 150)
        Tref = best_wai_gen[0].Tref

        colormap = mpl.colormaps['crest'] # or 'plasma', 'coolwarm', etc.
        norm = colors.Normalize(vmin=0, vmax=len(best_wai_gen))

        plt.figure(figsize=(5, 4))
        plt.xlabel("$p/p_0[T]$")
        plt.xticks(p_ovr_p0_ticks)
        plt.ylabel(
            f"water adsorption at {wais[0].Tref:.0f}°C\n[kg H$_2$O/kg sorbent]"
        )
        for g in range(len(best_wai_gen)):
            plt.plot(
                p_over_p0s, 
                [
                    best_wai_gen[g].water_ads(Tref, p_over_p0) 
                    for p_over_p0 in p_over_p0s
                ], 
                color=colormap(norm(g))
            )

        sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
        # sm.set_array([]) # Required for matplotlib versions < 3.4
        cb_ax = plt.gca().inset_axes(
            [0.7, 0.2, 0.2, 0.6]
        )
        cb_ax.axis("off")
        plt.colorbar(
            sm, ax=cb_ax, label='generation', 
        )
        plt.xlim([0, 1])
        plt.ylim([0, 0.5])

        plt.tight_layout()
        plt.savefig(
            weather.tag + "wai_progress.pdf", format="pdf"
        )
        plt.show()

    viz_best_wais(best_wai_gen)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### analyze optimal isotherm
    """)
    return


@app.cell
def _(weather):
    weather.ads_des_conditions
    return


@app.cell
def _(best_wai, os, pickle, weather):
    pf_name = "pkls/" + weather.tag + '_opt_isotherm.pkl'
    os.makedirs("pkls", exist_ok=True)
    with open(pf_name, 'wb') as pf:
        pickle.dump(best_wai, pf)
        print("saved in: ", pf_name)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### inspect day-to-day performance
    """)
    return


@app.cell
def _(
    best_wai,
    plt,
    score_fitness,
    set_weather_cols_axis,
    short_to_proper_weather_cols,
    sns,
    weather,
    weather_cols,
):
    def viz_daily_performance(best_wai, weather):
        scores, fitness = score_fitness(best_wai, weather)

        performance_data = weather.ads_des_conditions.copy()

        cols_to_plot = weather_cols + ["water delivery [kg H$_2$O/kg MOF]"]

        # Initialize the grid
        pp = sns.PairGrid(
            performance_data.rename(
                columns=short_to_proper_weather_cols
            ),
            vars=[short_to_proper_weather_cols[w] for w in weather_cols] + ["water del [kg H$_2$O/kg MOF]"],
            hue="water del [kg H$_2$O/kg MOF]", 
            corner=True
        )

        # Map only to the off-diagonal (lower) plots
        pp.map_lower(sns.scatterplot)

        # Optional: Add a legend since we are using PairGrid manually
        handles, labels = pp.axes[1, 0].get_legend_handles_labels()

        pp.fig.legend(
            handles, 
            labels,
            title="water del [kg H$_2$O/kg MOF]",
            loc="upper right", 
            bbox_to_anchor=(0.8, 0.8) 
        )

        set_weather_cols_axis(pp)

        for i in range(5):
            pp.axes[i, i].set_visible(False)

        plt.savefig(
            weather.tag + "daily_performance.pdf", format="pdf",
            bbox_inches="tight"
        )

        plt.show()

    viz_daily_performance(best_wai, weather)
    return


@app.cell
def _(T_range, colors, mpl, np, p_ovr_p0_ticks, plt):
    def viz_water_del(wai, weather, date, savename=""):
        day_data = weather.ads_des_conditions[
            weather.ads_des_conditions["date"].apply(
                lambda d: d.date() == date
            )
        ].iloc[0, :]

        p_over_p0s = np.linspace(0, 1.0, 100)

        fig = plt.figure()
        plt.xlabel("relative humidity $p / [p_0(T)]$")
        plt.xticks(p_ovr_p0_ticks)
        plt.xlim(0, 1.0)
        plt.ylabel("water adsorption\n[kg H$_2$O/kg sorbent]")

        colormap = mpl.colormaps['coolwarm'] # or 'plasma', 'coolwarm', etc.
        norm = colors.Normalize(vmin=T_range[0], vmax=T_range[1])

        # capture conditions
        T_night, p_ovr_p0_night = day_data["ads T [°C]"], day_data["ads P/P0"]
        w_night = wai.water_ads(T_night, p_ovr_p0_night)

        # release conditions
        T_day, p_ovr_p0_day = day_data["des T [°C]"], day_data["des P/P0"]
        w_day = wai.water_ads(T_day, p_ovr_p0_day)

        # viz capture and release conditions on the two isotherms
        for T, p_ovr_p0, w, label in [
            [T_night, p_ovr_p0_night, w_night, "capture state"],
            [T_day, p_ovr_p0_day, w_day, "release state"],
        ]:
            plt.plot(
                p_over_p0s, 
                [wai.water_ads(T, p_over_p0) for p_over_p0 in p_over_p0s],
                color=colormap(norm(T)),
                label=f"T = {T:0.1f}°C",
                lw=3, clip_on=False
            )
            plt.scatter(
                p_ovr_p0, w,
                color=colormap(norm(T)), label=label, zorder=25,
                marker="*", 
                edgecolor="black",
                s=150, clip_on=False
            )

        # put water delivery there
        plt.arrow(p_ovr_p0_night, w_night, 0, w_day - w_night, 
              color="black", head_width=0.008, length_includes_head=True)
        plt.text(
            p_ovr_p0_night + 0.01, (w_day + w_night) / 2,
            f" water delivery:\n {w_night-w_day:0.2f} kg/kg",
            color='black', 
            fontsize=10, 
            verticalalignment='center'
        )
        plt.plot(
            [p_ovr_p0_night, p_ovr_p0_day], [w_day, w_day], 
            color="gray", linestyle="--"
        )

        plt.legend(fontsize=12, title=date)
        plt.xlim([0, 1])
        plt.ylim(ymin=0.0)

        if not savename == "":
            plt.savefig(
                weather.tag + savename + ".pdf", format="pdf", bbox_inches="tight"
            )
        plt.show()
    return (viz_water_del,)


@app.cell
def _(best_wai, datetime, viz_water_del, weather):
    viz_water_del(best_wai, weather, datetime.date(2025, 7, 1))
    return


@app.cell(hide_code=True)
def _(mo):
    failure_explorer = mo.ui.slider(
        start=0, stop=25, label="failure ID"
    )
    failure_explorer
    return (failure_explorer,)


@app.cell
def _(best_wai, weather):
    attach_water_delivery(best_wai, weather)
    failure_list = weather.ads_des_conditions.copy().sort_values("water del [kg H$_2$O/kg MOF]")
    failure_list
    return (failure_list,)


@app.cell
def _(best_wai, failure_explorer, failure_list, viz_water_del, weather):
    viz_water_del(best_wai, weather, failure_list.iloc[failure_explorer.value]["date"].date())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## baseline: a stepped adsorption isotherm
    search for best stepped adsorption isotherm
    """)
    return


@app.cell
def _(WaterAdsorptionIsotherm, n, np, score_fitness, weather):
    def search_step_wais(dim):
        wais = [WaterAdsorptionIsotherm(dim) for i in range(dim-1)]

        for i_step in np.arange(1, dim):
            wais[i_step-1].endow_stepped_isotherm(i_step)

        fitnesses = np.array(
            [score_fitness(wai, weather)[1] for wai in wais]
        )
        id_opt = np.argmax(fitnesses)
        opt_fitness = np.max(fitnesses)
        best_wai_step = wais[id_opt]
        best_wai_step.label = "optimal step"
        return wais, fitnesses, id_opt, best_wai_step, opt_fitness

    step_wais, step_fitnesses, id_opt_step, best_wai_step, best_fitness_step = search_step_wais(n)
    return (
        best_fitness_step,
        best_wai_step,
        id_opt_step,
        step_fitnesses,
        step_wais,
    )


@app.cell
def _(best_wai, best_wai_step, compare_wais, weather):
    compare_wais([best_wai, best_wai_step], weather)
    return


@app.cell
def _(best_wai_step):
    best_wai_step.get_p_ovr_p0_half_max(verbose=True)
    return


@app.cell
def _(best_wai_step, viz_monthly_water_del, weather):
    viz_monthly_water_del(best_wai_step, weather)
    return


@app.cell
def _(colors, id_opt_step, mpl, np, plt, step_fitnesses, step_wais, weather):
    def viz_step_wais(step_wais, step_fitnesses, id_opt_step):
        Tref = step_wais[0].Tref
        w_max = step_wais[0].w_max

        p_over_p0s = np.linspace(0, 1.0, 100)

        plt.figure()

        plt.xlabel("relative humidity $p / [p_0(T)]$")
        plt.ylabel("water adsorption [kg H$_2$O/kg sorbent]")

        colormap = mpl.colormaps['viridis'] # or 'plasma', 'coolwarm', etc.
        norm = colors.Normalize(vmin=0.0, vmax=np.max(step_fitnesses))

        for i in range(len(step_wais)):
            plt.plot(
                p_over_p0s, 
                [step_wais[i].water_ads(Tref, p_over_p0) for p_over_p0 in p_over_p0s],
                color=colormap(norm(step_fitnesses[i])),
                clip_on=False
            )

        sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
        plt.colorbar(sm, ax=plt.gca(), label='fitness [kg H$_2$O/kg MOF]')
        plt.xlim(0, 1.0)
        plt.ylim(0, w_max)

        plt.savefig(
            weather.tag + "step_search.pdf",
            format="pdf",  bbox_inches="tight"
        )

        plt.show()

    viz_step_wais(step_wais, step_fitnesses, id_opt_step)
    return


@app.cell
def _(best_fitness, best_fitness_step):
    print(
        "% mass savings over a step: ",
        (best_fitness - best_fitness_step) / best_fitness_step * 100
    )
    return


@app.cell
def _(best_wai_step):
    best_wai_step.draw()
    return


@app.cell
def _(best_wai_step, weather):
    attach_water_delivery(best_wai_step, weather)
    failure_list_step = weather.ads_des_conditions.copy().sort_values("water del [kg H$_2$O/kg MOF]")
    failure_list_step
    return (failure_list_step,)


@app.cell
def _(mo):
    step_failure_explorer = mo.ui.slider(
        start=0, stop=25, label="failure ID"
    )
    step_failure_explorer
    return (step_failure_explorer,)


@app.cell
def _(
    best_wai_step,
    failure_list_step,
    step_failure_explorer,
    viz_water_del,
    weather,
):
    viz_water_del(
        best_wai_step, weather, 
        failure_list_step.iloc[step_failure_explorer.value]["date"].date(),
        # savename="failure_left"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # how does the opt isotherm from another city translate to here?
    """)
    return


@app.cell
def _(weather):
    if "Yuma" in weather.tag:
        other_tag = "Riley"
    elif "Riley" in weather.tag:
        other_tag = "Yuma"
    elif "Stovepipe" in weather.tag:
        other_tag = "Riley"
    else:
        other_tag = "Riley"
    other_tag
    return (other_tag,)


@app.cell
def _(other_tag, pickle):
    other_pf_name = "pkls/" + other_tag + '_opt_isotherm.pkl'
    with open(other_pf_name, 'rb') as opf:
        best_wai_other_city = pickle.load(opf)
    best_wai_other_city
    return (best_wai_other_city,)


@app.cell
def _(weather):
    weather.tag
    return


@app.cell
def _(best_wai_other_city, get_performance_data, weather):
    opt_performance_nontailored = get_performance_data(
        best_wai_other_city, weather, w_low=0.2
    )
    return (opt_performance_nontailored,)


@app.cell
def _(opt_performance_nontailored):
    non_tailored_failures = opt_performance_nontailored.groupby("failure").get_group(True)
    non_tailored_failures
    return (non_tailored_failures,)


@app.cell
def _(mo, non_tailored_failures):
    non_tailor_failure_explorer = mo.ui.slider(
        start=0, stop=non_tailored_failures.shape[0] - 1, label="failure ID"
    )
    non_tailor_failure_explorer
    return (non_tailor_failure_explorer,)


@app.cell
def _(
    best_wai_other_city,
    non_tailor_failure_explorer,
    non_tailored_failures,
    viz_water_del,
    weather,
):
    if weather.location == "Riley":
        failure_id = 11
    elif weather.location == "Yuma":
        failure_id = 47
    else:
        failure_id = non_tailor_failure_explorer.value
        print(failure_id)

    viz_water_del(
        best_wai_other_city, weather, 
        non_tailored_failures.iloc[failure_id]["date"].date(),
        savename="other_city_failure"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # shape-matching
    """)
    return


if __name__ == "__main__":
    app.run()
