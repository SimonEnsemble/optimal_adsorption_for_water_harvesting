import marimo

__generated_with = "0.23.14"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import math
    import numpy as np
    import os
    import datetime
    import random
    import warnings
    from scipy.special import comb
    from scipy.stats import gaussian_kde
    import matplotlib.dates as mdates
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from matplotlib.ticker import MaxNLocator
    import matplotlib.colors as colors
    import seaborn as sns
    from aquarel import load_theme
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import pickle
    from scipy.stats import gaussian_kde

    theme = load_theme("umbra_light")
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
        MaxNLocator,
        ccrs,
        cfeature,
        colors,
        comb,
        datetime,
        gaussian_kde,
        mo,
        mpl,
        np,
        os,
        pd,
        pickle,
        plt,
        random,
        sns,
        warnings,
    )


@app.cell
def _(sns):
    my_colors = sns.color_palette("Set2")
    my_colors
    return (my_colors,)


@app.cell
def _(my_colors):
    idea_to_color =  {
        'day': my_colors[1], 
        "night": my_colors[2],
        "water ads": my_colors[0],
        "Yuma": my_colors[3],
        "Riley": my_colors[4],
        "Stovepipe": my_colors[6]
    }
    idea_to_color["ads"] = idea_to_color["night"]
    idea_to_color["des"] = idea_to_color["day"]
    return (idea_to_color,)


@app.cell
def _(os):
    fig_dir = "figs"
    os.makedirs(fig_dir, exist_ok=True)
    return (fig_dir,)


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
    T_range = [0.0, 70.0] # deg C

    # ticks for plots
    T_ticks = np.linspace(
        T_range[0], T_range[1], 
        int(np.ceil((T_range[1] - T_range[0]) / 10)) + 1
    )
    p_ovr_p0_ticks = np.linspace(0, 1, 11)
    return T_range, T_ticks, p_ovr_p0_ticks


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
    return (city_to_state,)


@app.cell
def _():
    city_to_coords = {
        'Tucson':    (-110.9742, 32.2540),
        'Socorro':   (-106.8914, 34.0584), 
        'Mercury':   (-115.9945, 36.6605), 
        'Stovepipe': (-117.1465, 36.6062),
        'Riley':     (-119.5038, 43.5415),
        'Yuma':      (-114.6277, 32.6927)
    }
    return (city_to_coords,)


@app.cell
def _(ccrs, cfeature, city_to_coords, plt):
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
        ax.set_extent([-125, -110, 30, 50])  # USA bounds

        for city in cities:
            lon = city_to_coords[city][0]
            lat = city_to_coords[city][1]
            ax.plot(lon, lat, marker="*", markersize=15, color="C0",
                    transform=ccrs.PlateCarree())
            ax.text(lon + 0.5, lat + 0.5, city, fontsize=10,
                    transform=ccrs.PlateCarree())
        savename = "figs/map"
        for city in cities:
            savename = savename + "_" + city
        plt.tight_layout()
        plt.savefig(savename + ".pdf", format="pdf")
        plt.show()

    return (viz_cities,)


@app.cell
def _(viz_cities):
    viz_cities(["Yuma", "Riley", "Stovepipe"])
    return


@app.cell
def _(np, os, pd):
    class WeatherData:
        """
        read in weather time series from a location in a given month and year
        """
        def __init__(
            self, location, month, year, 
            verbose=True, time_to_hour={'day': 15, 'night': 5}
        ):
            self.location = location
            self.month = month
            self.year = year
            if verbose:
                print(f"loc: {location}. month: {month}/{year}")
                print("\tnighttime adsorption hr: ", time_to_hour["night"])
                print("\tdaytime harvest hr: ",      time_to_hour["day"])

            self.relevant_weather_cols = [
                "T_HR_AVG", "RH_HR_AVG", "SUR_TEMP", "SUR_RH_HR_AVG"
            ]

            self.verbose = verbose
            self.raw_data = None
            self.ads_des_conditions = None

            self._read_raw_data()
            self._filter_missing()
            self._process_datetimes()
            self._remove_rainy_days()
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
            ] # keep only 2024

            # get hours
            self.raw_data["time"] = [
                pd.Timedelta(hours=h) for h in self.raw_data["LST_TIME"] / 100
            ]

            self.raw_data["datetime"] = (
                self.raw_data["date"] + self.raw_data["time"]
            )

            # filter by month
            self.raw_data = self.raw_data.loc[
                self.raw_data["datetime"].dt.month == self.month
            ]

        def _remove_rainy_days(self):
            self.raw_data["rain_daily_total"] = (
                self.raw_data.groupby("LST_DATE")["P_CALC"].transform("sum")
            )

            if self.verbose:
                n_rainy_days = (
                    self.raw_data.loc[
                        self.raw_data["rain_daily_total"] > 0, "LST_DATE"
                    ].nunique()
                )
                print(f"\tremoved {n_rainy_days} rainy days")

            self.raw_data = self.raw_data[
                self.raw_data["rain_daily_total"] == 0.0
            ]

        def _filter_missing(self):
            ids_bad = self.raw_data["T_HR_AVG"] < -999.0
            n_bad = np.sum(ids_bad)
            if n_bad > 0:
                print(f"\tfiltering {n_bad} missing rows in raw data")
                self.raw_data = self.raw_data[~ ids_bad]

        def _prune_raw_data(self):
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
            self.raw_data["SUR_RH_HR_AVG"] = self.raw_data.apply(
                lambda day: day["RH_HR_AVG"] * water_p0(day["T_HR_AVG"])
                    / water_p0(day["SUR_TEMP"]), 
                axis=1
            )

        def _attach_ads_des_conditions(self):
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
                    "day_SUR_RH_HR_AVG": 'des P/P0'
                }
            )
            for rh_col in ['des P/P0', 'ads P/P0']:
                self.ads_des_conditions[rh_col] = (
                    self.ads_des_conditions[rh_col] / 100.0
                )

            self.ads_des_conditions = self.ads_des_conditions[
                ['date', 'ads T [°C]', 'ads P/P0', 'des T [°C]', 'des P/P0']
            ]

            self.ads_des_conditions["location"] = self.location

        def sees_ice(self):
            return self.ads_des_conditions[
                ["ads T [°C]", "des T [°C]"]
            ].min().min() < 0.0

    return (WeatherData,)


@app.cell
def _(WeatherData):
    wdata = WeatherData("Yuma", 7, 2024)
    wdata.ads_des_conditions
    return


@app.cell
def _(T_range, idea_to_color, pd, plt):
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

            assert self.ads_des_conditions["date"].nunique() == n_rows

            self._assert_in_T_range()

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

    return (Weather,)


@app.cell
def _(weather):
    weather.viz_timeseries()
    return


@app.cell(hide_code=True)
def _(mo):
    dropdown = mo.ui.dropdown(
        options=["Yuma", "Riley", "Stovepipe", "Yuma & Riley & Stovepipe"], 
        value="Yuma", label="choose location"
    )
    dropdown
    return (dropdown,)


@app.cell
def _(Weather, WeatherData, dropdown, random):
    summer_months = [6, 7, 8] # meterological
    yrs = [2023, 2024, 2025]

    random_cities = ["Yuma", "Riley", "Stovepipe"]

    if dropdown.value in ["Yuma", "Riley", "Stovepipe"]:
        _city = dropdown.value
        weather_datas = [
            WeatherData(_city, mo, yr) 
            for mo in summer_months
            for yr in yrs
        ]
    elif dropdown.value == "Yuma & Riley & Stovepipe":
        weather_datas = []


        random.seed(42)
        for yr in yrs:
            _mos = summer_months.copy()
            for _i in range(len(_mos)):
                _mo = _mos.pop(random.randrange(len(_mos)))
                random.shuffle(random_cities)
                for _city in random_cities:
                    _weather_data = WeatherData(_city, _mo, yr)
                    if not _weather_data.sees_ice():
                        weather_datas.append(_weather_data)
                        break

    weather =  Weather(
        # list of weather data
        weather_datas,
        # tag
        dropdown.value
    )
    weather.ads_des_conditions
    return random_cities, weather


@app.cell
def _(weather):
    weather.viz_timeseries()
    return


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
    ccrs,
    cfeature,
    city_to_coords,
    idea_to_color,
    my_colors,
    p_ovr_p0_ticks,
    plt,
    random_cities,
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
            hue="location",
            hue_order=random_cities,
            palette=idea_to_color,
            corner=True,
            plot_kws=dict(marker="+", linewidth=1, color=my_colors[0]),
            diag_kws=dict(fill=False, color=my_colors[0]),
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
        fig = pp.fig
        fig.canvas.draw()

        # Infer the top-right 2×2 block from existing axes:
        #   columns 2 & 3 → use any row that has those axes (e.g. row 3)
        #   rows 0 & 1    → use the diagonal axes [0,0] and [1,1]
        x0 = pp.axes[3, 2].get_position().x0
        x1 = pp.axes[3, 3].get_position().x1
        y0 = pp.axes[1, 1].get_position().y0
        y1 = pp.axes[0, 0].get_position().y1

        map_ax = fig.add_axes(
            [x0, y0, x1 - x0, y1 - y0],
            projection=ccrs.PlateCarree()
        )
        map_ax.add_feature(cfeature.LAND)
        map_ax.add_feature(cfeature.OCEAN)
        map_ax.add_feature(cfeature.BORDERS, linewidth=0.5)
        map_ax.add_feature(cfeature.COASTLINE)
        map_ax.add_feature(cfeature.STATES, linewidth=0.5, edgecolor="gray")
        map_ax.set_extent([-125, -110, 30, 50])

        cities = ["Yuma", "Stovepipe", "Riley"]
        for city in cities:
            lon, lat = city_to_coords[city]
            map_ax.plot(
                lon, lat, marker="*", markersize=20, 
                color=idea_to_color[city], transform=ccrs.PlateCarree()
            )
            map_ax.text(
                lon + 0.5, lat + 0.5, city, 
                fontsize=14, transform=ccrs.PlateCarree()
            )

        plt.tight_layout()
        plt.savefig(
            weather.tag + "ads_des_conditions.pdf", 
            format="pdf"
        )
        plt.show()
    return set_weather_cols_axis, short_to_proper_weather_cols, weather_cols


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 🛏️ modeling a water adsorption isotherm in a MOF bed
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
            assert np.all(x >= 0.0)
            assert np.all(x <= 1.0)
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


@app.cell
def _(BernPolyBasis, colors, mpl, np, plt):
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

            plt.figure()

            plt.xlabel("relative humidity $p / [p_0(T)]$")
            plt.ylabel("water adsorption [kg H$_2$O/kg sorbent]")

            colormap = mpl.colormaps['coolwarm'] # or 'plasma', 'coolwarm', etc.
            norm = colors.Normalize(vmin=10.0, vmax=60.0)

            for T in np.linspace(0, 80, 6):
                plt.plot(
                    p_over_p0s, 
                    [self.water_ads(T, p_over_p0) for p_over_p0 in p_over_p0s],
                    color=colormap(norm(T)),
                    clip_on=False
                )

            sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
            plt.colorbar(sm, ax=plt.gca(), label='temperature [°C]')
            plt.xlim(0, 1.0)
            plt.ylim(0, self.w_max)

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


@app.cell
def _(np):
    # conditional value at risk
    def score_fitness(wai, weather, alpha=1/7 * 100):
        # get dist'n of water dels
        water_dels = wai.water_del(weather.ads_des_conditions)
        # get worst-case water delivery, ignoring alpha % of hard cases.
        #   (variance at risk)
        var = np.percentile(water_dels, alpha)
        return np.mean(water_dels[water_dels <= var])

    return (score_fitness,)


@app.cell
def _(score_fitness, wai, weather):
    fitness = score_fitness(wai, weather)
    fitness
    return (fitness,)


@app.cell
def _(gaussian_kde, np):
    def draw_fitness(
        ax, wdels, fitness, color, label, 
        alpha=1/7 * 100, orientation="vertical"
    ):
        var = np.percentile(wdels, alpha)

        kde = gaussian_kde(wdels)
        xs = np.linspace(0.0, max(wdels.max(), 0.5), 300)
        density = kde(xs)

        lo_mask = xs < var
        hi_mask = xs >= var

        if orientation == "vertical":
            ax.plot(xs, density, color=color, linewidth=1.5)
            ax.fill_between(
                xs[lo_mask], density[lo_mask], color=color, alpha=0.3
            )
            ax.fill_between(
                xs[hi_mask], density[hi_mask], color=color, alpha=0.1,
                label=label
            )
            ax.axvline(fitness, linestyle="--", color=color)
        else:
            ax.plot(density, xs, color=color, linewidth=1.5)
            ax.fill_betweenx(
                xs[lo_mask], density[lo_mask], color=color, alpha=0.4
            )
            ax.fill_betweenx(
                xs[hi_mask], density[hi_mask], color=color, alpha=0.1,
                label=label
            )
            ax.axhline(fitness, linestyle="--", color=color)

    return (draw_fitness,)


@app.cell
def _(draw_fitness, fitness, my_colors, plt, wai, weather):
    plt.figure()
    draw_fitness(
        plt.gca(),
        wai.water_del(weather.ads_des_conditions), 
        fitness, 
        my_colors[4],
        ""
    )
    plt.xlabel("water delivery [kg H$_2$O/kg sorbent]")
    plt.ylabel("# days")
    plt.tight_layout()

    plt.savefig(weather.tag + "eg_var.pdf", format="pdf")
    plt.show()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 🎲 random WAIs to explore
    """)
    return


@app.cell
def _(draw_rh_distn, my_colors, np, p_ovr_p0_ticks, plt, score_fitness):
    def compare_wais(wais, weather, savetag=""):
        the_colors = [my_colors[0]] + my_colors[3:]
        p_over_p0s = np.linspace(0, 1.0, 100)

        fig = plt.figure(figsize=(6.25, 5), layout="constrained")
        gs = fig.add_gridspec(2, 2, height_ratios=[1, 3], width_ratios=[2, 1])
        ax00 = fig.add_subplot(gs[0, 0])
        ax10 = fig.add_subplot(gs[1, 0], sharex=ax00) # Only these two share
        ax01 = fig.add_subplot(gs[0, 1])
        ax11 = fig.add_subplot(gs[1, 1], sharey=ax10)
        axs = np.array([[ax00, ax01],
                        [ax10, ax11]])

        axs[0, 1].axis('off')

        ###
        #   adsorption isotherm
        ###
        axs[1, 0].set_xlabel("$p / [p_0(T)]$")
        axs[0, 0].tick_params(axis='x', labelbottom=False)
        axs[1, 0].set_xticks(p_ovr_p0_ticks)
        axs[1, 0].tick_params(axis='x', labelrotation=90)
        axs[1, 0].set_ylabel(
            f"water adsorption at {wais[0].Tref:.0f}°C\n[kg H$_2$O/kg sorbent]"
        )

        for w, wai in enumerate(wais):
            axs[1, 0].plot(
                p_over_p0s, 
                [wai.water_ads(wai.Tref, p_over_p0) for p_over_p0 in p_over_p0s],
                color=the_colors[w],
                label=f"#{w}"
            )

        axs[1, 0].set_xlim(0, 1)
        axs[1, 0].set_ylim(0, wais[0].w_max)
        axs[1, 0].legend(title="material", fontsize=8, title_fontsize=10)

        ###
        #   P/P0 distns
        ###
        draw_rh_distn(axs[0, 0], weather)

        ###
        #   working cap dist'n
        ###
        bins = np.linspace(0, wais[0].w_max, 12)
        for w, wai in enumerate(wais):
            fitness = score_fitness(wai, weather)

            axs[1, 1].hist(
                wai.water_del(weather.ads_des_conditions),
                orientation='horizontal', 
                edgecolor=the_colors[w], histtype='step',
                bins=bins
            )
            axs[1, 1].hist(
                wai.water_del(weather.ads_des_conditions),
                orientation='horizontal', 
                color=the_colors[w], alpha=0.25,
                bins=bins
            )

            axs[1, 1].axhline(
                fitness, color=the_colors[w], linestyle="--"
            )
        axs[1, 1].set_xlabel("# days")
        axs[1, 1].set_xticks([0, 100, 200])
        axs[1, 1].set_yticks([0.1 * i for i in range(6)])
        axs[1, 1].set_xlim(0, 300)
        axs[1, 1].set_ylabel("water delivery [kg H$_2$O/kg MOF]")
        # axs[1, 1].legend(fontsize=12)

        # fitness label:
        fitness_label = f"fitness:\n{fitness:.2f} kg H$_2$O/kg MOF",

        plt.savefig(
            weather.tag + "compare" + savetag + ".pdf",
            format="pdf",  bbox_inches="tight"
        )

        plt.show()

    return (compare_wais,)


@app.cell
def _(WaterAdsorptionIsotherm, compare_wais, np, score_fitness, weather):
    _wais = [WaterAdsorptionIsotherm(10) for i in range(51)]
    [wai.endow_random_isotherm() for wai in _wais]

    _fitness = [score_fitness(wai, weather) for wai in _wais]
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

        fitness = score_fitness(wai, weather)
        if verbose:
            print("---local search---")
            print("current fitness: ", fitness)

        # max out capacity at high p/p0 until fitness decreases
        for i in range(1, wai.n): # walk backwards thru array
            new_wai.bs[-i:] = wai.w_max
            new_fitness = score_fitness(new_wai, weather)
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
            new_fitness = score_fitness(new_wai, weather)
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
        fitnesses = np.array([score_fitness(wai, weather) for wai in wais])

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

    fitnesses = np.array([score_fitness(wai, weather) for wai in wais])

    # second generation
    new_wais = evolve(wais, weather, n_elite=5)
    new_fitnesses = np.array(
        [score_fitness(new_wai, weather) for new_wai in new_wais]
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


@app.cell
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
        fitnesses = np.array([score_fitness(wai, weather) for wai in wais])

        # store progress
        fitnesses_gen = [fitnesses]
        best_wai_gen = [wais[np.argmax(fitnesses)]]

        # evolve over generations
        for g in range(1, n_generations):
            wais = evolve(wais, weather)
            fitnesses = np.array([score_fitness(wai, weather) for wai in wais])

            fitnesses_gen.append(fitnesses)
            best_wai_gen.append(wais[np.argmax(fitnesses)])

        best_wai = wais[np.argmax(fitnesses)]
        best_fitness = np.max(fitnesses)

        return fitnesses_gen, best_wai_gen, best_wai, best_fitness

    return (do_evolution,)


@app.cell
def _(do_evolution, run_evol_cbox, weather):
    pop_size = 50
    n_generations = 25
    n = 40
    if run_evol_cbox.value:
        fitnesses_gen, best_wai_gen, best_wai, best_fitness = do_evolution(
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### analyze progress
    """)
    return


@app.cell
def _(fitnesses_gen, pd, plt, sns, weather):
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
            ax=ax
        )
        plt.tick_params(axis='x', labelrotation=90)
        # plt.axhline(
        #     y=step_fitnesses[id_opt_step], 
        #     color="gray", linestyle="--", zorder=-1
        # )
        plt.tight_layout()
        plt.savefig(
            weather.tag + "fitness_progress.pdf", format="pdf"
        )
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


@app.function
def get_performance_data(wai, weather, w_low=0.15):
    perf_data = weather.ads_des_conditions.copy()
    perf_data["water del [kg H$_2$O/kg MOF]"] = wai.water_del(
        weather.ads_des_conditions
    )
    perf_data["failure"] = perf_data.apply(
        lambda row: row["water del [kg H$_2$O/kg MOF]"] < w_low, axis=1
    )
    print(
        f"# failures: ", 
        perf_data["failure"].sum(), " / ", perf_data.shape[0]
    )
    return perf_data


@app.cell
def _(best_wai, weather):
    opt_performance_data = get_performance_data(best_wai, weather, w_low=0.1)
    opt_performance_data
    return (opt_performance_data,)


@app.cell
def _(MaxNLocator, gaussian_kde, idea_to_color, np):
    def draw_rh_distn(ax, weather):
        xs = np.linspace(0, 1, 200)

        kde_des = gaussian_kde(weather.ads_des_conditions["des P/P0"])
        ax.plot(xs, kde_des(xs), color=idea_to_color["night"], linewidth=1.5)
        ax.fill_between(
            xs, kde_des(xs), color=idea_to_color["night"], alpha=0.25,
            label="release"
        )

        kde_ads = gaussian_kde(weather.ads_des_conditions["ads P/P0"])
        ax.plot(xs, kde_ads(xs), color=idea_to_color["day"], linewidth=1.5)
        ax.fill_between(
            xs, kde_ads(xs), color=idea_to_color["day"], alpha=0.25,
            label="capture"
        )

        ax.set_ylabel("density")
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
        ax.set_xlim(0, 1)
        ax.set_ylim(ymin=0.0)
        ax.legend(fontsize=12)

    return (draw_rh_distn,)


@app.cell
def _(
    MaxNLocator,
    T_range,
    T_ticks,
    colors,
    draw_fitness,
    draw_rh_distn,
    mpl,
    my_colors,
    np,
    p_ovr_p0_ticks,
    plt,
    score_fitness,
):
    def draw_opt(best_wai, weather, savetag=""):
        p_over_p0s = np.linspace(0, 1.0, 100)

        # fig, axs = plt.subplots(
        #     2, 2, 
        #     gridspec_kw={'height_ratios': [1, 3], 'width_ratios': [2, 1]},
        #     figsize=(5, 7),
        #     layout="constrained"
        # )
        fig = plt.figure(figsize=(7, 5), layout="constrained")
        gs = fig.add_gridspec(2, 2, height_ratios=[1, 3], width_ratios=[2, 1])
        ax00 = fig.add_subplot(gs[0, 0])
        ax10 = fig.add_subplot(gs[1, 0], sharex=ax00) # Only these two share
        ax01 = fig.add_subplot(gs[0, 1])
        ax11 = fig.add_subplot(gs[1, 1], sharey=ax10)
        axs = np.array([[ax00, ax01],
                        [ax10, ax11]])

        axs[0, 1].axis('off')
        # axs[1, 0].get_shared_x_axes().join(axs[1, 0], axs[0, 0])

        ###
        #   adsorption isotherm
        ###
        axs[1, 0].set_xlabel("relative humidity $p / [p_0(T)]$")
        axs[0, 0].tick_params(axis='x', labelbottom=False)
        axs[1, 0].set_xticks(p_ovr_p0_ticks)
        axs[1, 0].tick_params(axis='x', labelrotation=90)
        axs[1, 0].set_ylabel("water adsorption [kg H$_2$O/kg sorbent]")

        colormap = mpl.colormaps['coolwarm'] # or 'plasma', 'coolwarm', etc.
        norm = colors.Normalize(vmin=T_range[0], vmax=T_range[1])

        for T in np.linspace(T_range[0], T_range[1], 4):
            axs[1, 0].plot(
                p_over_p0s, 
                [best_wai.water_ads(T, p_over_p0) for p_over_p0 in p_over_p0s],
                color=colormap(norm(T))
            )

        sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
        # sm.set_array([]) # Required for matplotlib versions < 3.4
        cb_ax = axs[1, 0].inset_axes(
            [0.5, 0.15, 0.2, 0.6]
        )
        cb_ax.axis("off")
        plt.colorbar(
            sm, ax=cb_ax, label='temperature [°C]', 
            ticks=T_ticks
            # orientation="horizontal"
        )
        axs[1, 0].set_xlim(0, 1.0)
        axs[1, 0].set_ylim(0, best_wai.w_max)

        ###
        #   P/P0 distns
        ###
        draw_rh_distn(axs[0, 0], weather)
        # axs[0, 0].set_yticks([0])
        axs[0, 0].yaxis.set_major_locator(MaxNLocator(nbins=3, integer=True))

        ###
        #   working cap dist'n
        ###
        fitness = score_fitness(best_wai, weather)
        print("fitness: ", fitness)

        draw_fitness(
            axs[1, 1], best_wai.water_del(weather.ads_des_conditions), 
            fitness, my_colors[4], "", orientation="horizontal"
        )

        axs[1, 1].set_xlabel("# days")
        # axs[1, 1].set_xticks([0])
        axs[1, 1].xaxis.set_major_locator(MaxNLocator(nbins=3, integer=True))
        axs[1, 1].set_xlim(xmin=0.0)
        axs[1, 1].set_ylabel("water delivery [kg H$_2$O/kg sorbent]")

        ###
        #   info
        ###
        # fitness label:
        fitness_label = f"{weather.tag}\nfitness:\n  {fitness:.2f} kg/kg"
        axs[0, 1].text(
            0.0, 0.5,                    # x, y in axes coordinates (0–1)
            fitness_label,
            transform=axs[0, 1].transAxes,
            verticalalignment='center',
            horizontalalignment='left',
            fontsize=10
        )

        plt.savefig(
            weather.tag + "best_wai_rich" + savetag + ".pdf",
            format="pdf",  bbox_inches="tight"
        )

        plt.show()

    return (draw_opt,)


@app.cell
def _(best_wai, pickle, weather):
    pf_name = weather.tag + 'opt_isotherm.pkl'
    with open(pf_name, 'wb') as pf:
        pickle.dump(best_wai, pf)
        print("saved in: ", pf_name)
    return


@app.cell
def _(best_wai, draw_opt, weather):
    draw_opt(best_wai, weather)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### inspect day-to-day performance
    """)
    return


@app.cell
def _(
    opt_performance_data,
    plt,
    set_weather_cols_axis,
    short_to_proper_weather_cols,
    sns,
    weather,
    weather_cols,
):
    def viz_daily_performance(performance_data):
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

    viz_daily_performance(opt_performance_data)
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

        fig = plt.figure(figsize=(4, 3.5))
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
                lw=2
            )
            plt.scatter(
                p_ovr_p0, w,
                color=colormap(norm(T)), label=label, zorder=25,
                marker="*", 
                edgecolor="black",
                s=150
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
def _(failures, mo):
    failure_explorer = mo.ui.slider(
        start=0, stop=failures.shape[0]-1, label="failure ID"
    )
    failure_explorer
    return (failure_explorer,)


@app.cell
def _(opt_performance_data):
    failures = opt_performance_data.groupby("failure").get_group(True)
    failures
    return (failures,)


@app.cell
def _(best_wai, failure_explorer, failures, viz_water_del, weather):
    viz_water_del(best_wai, weather, failures.iloc[failure_explorer.value]["date"].date())
    return


@app.cell
def _(opt_performance_data):
    opt_performance_data
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
            [score_fitness(wai, weather) for wai in wais]
        )
        id_opt = np.argmax(fitnesses)
        opt_fitness = np.max(fitnesses)
        best_wai_step = wais[id_opt]
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
def _(
    best_wai,
    best_wai_step,
    draw_fitness,
    my_colors,
    np,
    plt,
    score_fitness,
    weather,
):
    def compare_fitnesses_step(weather, best_wai, best_wai_step):
        fitness = score_fitness(best_wai, weather)
        print("fitness: ", fitness)

        fitness_step = score_fitness(best_wai_step, weather)
        print("fitness with step: ", fitness_step)

        wdels = best_wai.water_del(weather.ads_des_conditions)
        print("mean water del: ", np.mean(wdels))
        wdels_step = best_wai_step.water_del(weather.ads_des_conditions)
        print("mean water del with best step WAI: ", np.mean(wdels_step))

        fig = plt.figure(figsize=(6, 3))
        plt.xlabel("water delivery [kg H$_2$O/kg sorbent]")
        plt.ylabel("# days")

        draw_fitness(plt.gca(), wdels, fitness, my_colors[4], label="optimal")
        draw_fitness(
            plt.gca(), wdels_step, fitness_step, 
            my_colors[6], label="optimal step"
        )
        # plt.title("water delivery distribution in " + weather.tag[:-1])
        plt.legend(title="water adsorption isotherm")

        plt.tight_layout()
        plt.savefig(
            weather.tag + "step_comparison.pdf", format="pdf"
        )

        plt.show()

    compare_fitnesses_step(weather, best_wai, best_wai_step)
    return


@app.cell
def _(best_wai, best_wai_step, my_colors, np, p_ovr_p0_ticks, plt, weather):
    def compare_opt_wai_and_opt_step_wai(best_wai, best_wai_step):
        plt.figure(figsize=(4, 4))

        plt.xlabel("relative humidity $p / [p_0(T)]$")
        plt.xticks(p_ovr_p0_ticks)
        plt.ylabel(
            f"water adsorption at {best_wai.Tref:.0f}°C\n[kg H$_2$O/kg sorbent]"
        )

        p_over_p0s = np.linspace(0, 1, 200)
        plt.plot(
            p_over_p0s, 
            [best_wai.water_ads(best_wai.Tref, p_over_p0) for p_over_p0 in p_over_p0s],
            color=my_colors[4], label="optimal", clip_on=False, lw=2
        )
        plt.plot(
            p_over_p0s, 
            [best_wai_step.water_ads(best_wai.Tref, p_over_p0) 
             for p_over_p0 in p_over_p0s
            ],
            color=my_colors[6], label="optimal step", clip_on=False, lw=2
        )
        plt.legend()

        plt.xlim(0, 1.0)
        plt.ylim(0, best_wai.w_max)

        plt.savefig(
            weather.tag + "step_vs_opt_wai.pdf",
            format="pdf",  bbox_inches="tight"
        )

        plt.show()

    compare_opt_wai_and_opt_step_wai(best_wai, best_wai_step)
    return


@app.cell
def _(best_wai_step):
    best_wai_step.get_p_ovr_p0_half_max(verbose=True)
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
def _(best_wai_step, draw_opt, weather):
    draw_opt(best_wai_step, weather, savetag="baseline")
    return


@app.cell
def _(best_wai_step, weather):
    opt_performance_step = get_performance_data(best_wai_step, weather, w_low=0.05)
    opt_performance_step
    return (opt_performance_step,)


@app.cell
def _(opt_performance_step):
    step_failures = opt_performance_step.groupby("failure").get_group(True)
    step_failures
    return (step_failures,)


@app.cell
def _(mo, step_failures):
    step_failure_explorer = mo.ui.slider(
        start=0, stop=step_failures.shape[0] - 1, label="failure ID"
    )
    step_failure_explorer
    return (step_failure_explorer,)


@app.cell
def _(step_failure_explorer, step_failures):
    step_failures.iloc[step_failure_explorer.value]["date"].date()
    return


@app.cell
def _(
    best_wai_step,
    step_failure_explorer,
    step_failures,
    viz_water_del,
    weather,
):
    viz_water_del(
        best_wai_step, weather, 
        step_failures.iloc[step_failure_explorer.value]["date"].date(),
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
        other_city = "Riley"
    elif "Riley" in weather.tag:
        other_city = "Yuma"
    elif "Stovepipe" in weather.tag:
        other_city = "Riley"
    else:
        other_city = "Yuma"
    other_city
    return (other_city,)


@app.cell
def _(fig_dir, other_city, pickle, weather):
    with open(fig_dir + f'/{other_city} (May-Sep.)/opt_isotherm.pkl', 'rb') as opf:
        weather.tag + 'opt_isotherm.pkl'
        best_wai_other_city = pickle.load(opf)
    return (best_wai_other_city,)


@app.cell
def _(
    best_wai,
    best_wai_other_city,
    city_to_state,
    draw_fitness,
    my_colors,
    np,
    other_city,
    plt,
    score_fitness,
    weather,
):
    def compare_fitnesses(weather, best_wai, best_wai_other_city):
        fitness = score_fitness(best_wai, weather)
        print("fitness: ", fitness)
        fitness_other_city = score_fitness(best_wai_other_city, weather)
        print("fitness with other city's WAI: ", fitness_other_city)

        wdels = best_wai.water_del(weather.ads_des_conditions)
        print("mean water del: ", np.mean(wdels))
        wdels_other_city = best_wai_other_city.water_del(weather.ads_des_conditions)
        print("mean water del with other city's WAI: ", np.mean(wdels_other_city))

        fig = plt.figure(figsize=(6, 3))
        plt.xlabel("water delivery [kg H$_2$O/kg sorbent]")
        plt.ylabel("# days")
        draw_fitness(wdels, fitness, my_colors[4], label=weather.tag[:-1])
        draw_fitness(
            wdels_other_city, fitness_other_city, my_colors[6], 
            label=other_city + f", {city_to_state[other_city]}"
        )
        # plt.title("water delivery distribution in " + weather.tag[:-1])
        plt.legend(title="isotherm optimized for...")
        plt.yticks([0])

        plt.tight_layout()
        plt.savefig(
            weather.tag + "other_city_isotherm_fitness.pdf", format="pdf"
        )

        plt.show()

    compare_fitnesses(weather, best_wai, best_wai_other_city)
    return


@app.cell
def _(weather):
    weather.tag
    return


@app.cell
def _(best_wai_other_city, weather):
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
