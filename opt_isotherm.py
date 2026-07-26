import marimo

__generated_with = "0.23.14"
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
    from scipy.optimize import minimize
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

    theme = load_theme("scientific")
    theme.set_transforms(trim=True)
    theme.apply()
    figsize = [6.4*0.875, 4.8*0.875]
    plt.rcParams.update(
        {
            'font.size': 14,
            'axes.titleweight': 'normal',
            'figure.titleweight': 'normal',
            'figure.figsize': figsize
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
        figsize,
        gaussian_kde,
        inset_axes,
        matplotlib,
        minimize,
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
        "Stovepipe": theme_colors[2],
        "Socorro": theme_colors[1],
        "Utqiagvik": theme_colors[4],
        "Yuma": my_colors[-1],
        "fitness": theme_colors[-1],
        "step": my_colors[-1],
        "mix": my_colors[4]
    }
    idea_to_color["ads"] = idea_to_color["night"]
    idea_to_color["des"] = idea_to_color["day"]
    return (idea_to_color,)


@app.cell
def _(idea_to_color, mixed_locations, sns):
    sns.color_palette([idea_to_color[city] for city in mixed_locations])
    return


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

    # viz_water_p0()
    return


@app.cell
def _():
    toy_RH_ambient = 0.3
    toy_T_ambient = 25.0
    toy_T_land = 40.0
    toy_RH_ambient * water_p0(toy_T_ambient) / water_p0(toy_T_land), 
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
def _():
    city_to_desert = {
        "Socorro": "Chihuahuan",
        "Stovepipe": "Mojave",
        "Riley": "OR High",
        "Utqiagvik": "AK Polar",
        "mix": "mix"
    }
    return (city_to_desert,)


@app.cell
def _(ccrs, cfeature, city_to_coords, city_to_desert, idea_to_color, plt):
    def viz_cities(
        cities, city_AB=None, savename=None, 
        extent=[-168, -99, 30, 71],
        xy_shift = {
            "Riley": [0, 5.5],
            "Stovepipe": [-7.75, 0.0],
            "Socorro": [0, 5.5],
            "Utqiagvik": [0, -5.5],
            "Yuma": [-7.5, 0]
        }
    ):


        fig, ax = plt.subplots(
            subplot_kw={"projection": ccrs.PlateCarree()}
        )

        # Add map features
        ax.add_feature(cfeature.LAND)
        ax.add_feature(cfeature.OCEAN)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5)
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.STATES, linewidth=0.5, edgecolor="gray")
        ax.set_extent(extent)  # USA bounds

        for city in cities:
            lon = city_to_coords[city][0]
            lat = city_to_coords[city][1]
            ax.plot(
                lon, lat, marker="*", markersize=15, color=idea_to_color[city],
                transform=ccrs.PlateCarree()
            )
            ax.text(
                lon + xy_shift[city][0], lat + xy_shift[city][1], city_to_desert[city] + "\nDesert", 
                fontsize=14, ha="center", va="center",
                transform=ccrs.PlateCarree(), 
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", boxstyle="round,pad=0.2")
            )

        if city_AB:
            lon1, lat1 = city_to_coords[city_AB[0]]
            lon2, lat2 = city_to_coords[city_AB[1]]
            ax.annotate(
                "",
                xy=(lon2, lat2), xycoords=ccrs.PlateCarree()._as_mpl_transform(ax),
                xytext=(lon1, lat1), textcoords=ccrs.PlateCarree()._as_mpl_transform(ax),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color="black",
                    lw=2,
                    shrinkA=0, shrinkB=0,  # avoid overlapping the star markers
                    mutation_scale=20
                )
            )

        # plt.tight_layout()
        if savename:
            plt.savefig(savename + ".pdf", format="pdf", bbox_inches="tight", pad_inches=0)
        plt.show()

    return (viz_cities,)


@app.cell
def _(mixed_locations, viz_cities):
    viz_cities(mixed_locations, savename="map_all")
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
            self.time_to_hour["ads"] = self.time_to_hour["night"]
            self.time_to_hour["des"] = self.time_to_hour["day"]
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
                'des T [°C]', 'des P/P0'
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
                wdata[time]["date"] = wdata[time]["datetime"].dt.normalize() # remove hour

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

            self.ads_des_conditions = self.ads_des_conditions[cols_to_put]

            self.ads_des_conditions["location"] = self.location

            # for a reasonable average... missing data otherwise.
            assert self.ads_des_conditions["date"].nunique() > 25

            # for quickly computing rolling sums
            self.ads_des_conditions["month_period"] = self.ads_des_conditions["date"].dt.to_period("M")
            self.ads_des_conditions["day"] = self.ads_des_conditions["date"].dt.day
            self.ads_des_conditions["date"] = self.ads_des_conditions["date"].dt.date

        def viz_timeseries(
            self, save=False, incl_legend=True, 
            legend_dx=0.0, legend_dy=0.0, savename=None
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
                    self.ads_des_conditions["date"] + pd.Timedelta(hours=self.time_to_hour[ads_des]), 
                    self.ads_des_conditions[f"{ads_des} T [°C]"],
                    edgecolors="black", clip_on=False,
                    marker="^", 
                    color=idea_to_color[ads_des], zorder=10, 
                    label=ads_des, 
                    s=25
                )

            axs[0].plot(
                self.raw_data["datetime"], self.raw_data["T_HR_AVG"], 
                label="bulk air", color="gray", linewidth=2
            )
            axs[0].plot(
                self.raw_data["datetime"], self.raw_data["SUR_TEMP"], 
                label="near-surface air", color="gray", linewidth=2, linestyle="--"
            )


            ###
            #   relative humidity
            ###
            axs[1].set_ylabel("relative\nhumidity")
            for ads_des in ["ads", "des"]:
                axs[1].scatter(
                    self.ads_des_conditions["date"] + pd.Timedelta(hours=self.time_to_hour[ads_des]), 
                    self.ads_des_conditions[f"{ads_des} P/P0"],
                    edgecolors="black", clip_on=False,
                    marker="v", 
                    color=idea_to_color[ads_des], zorder=10, 
                    label=ads_des, 
                    s=25
                )

                axs[1].plot(
                    self.raw_data["datetime"], self.raw_data["RH_HR_AVG"] / 100, 
                    color="gray"
                )
                axs[1].plot(
                    self.raw_data["datetime"], self.raw_data["SUR_RH_HR_AVG"] / 100, 
                    color="gray", linestyle="--"
                )

            axs[1].legend(
                    prop={'size': 10}, ncol=1, 
                    bbox_to_anchor=(0., 1.0 + legend_dy, 1.0 + legend_dx, .1),
                   loc="center left"
            )#, loc="center left")

            if savename:
                plt.tight_layout()
                plt.savefig(savename + ".pdf", format="pdf")
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

        def all_consecutive_days(self):
            gaps = self.ads_des_conditions["date"].diff().dropna()
            return (gaps == pd.Timedelta(days=1)).all()

    return (WeatherData,)


@app.cell
def _(wdata):
    wdata.ads_des_conditions["date"]
    return


@app.cell
def _(wdata):
    wdata.all_consecutive_days()
    return


@app.cell
def _(WeatherData):
    wdata = WeatherData("Stovepipe", [6], 2021)
    wdata.ads_des_conditions
    return (wdata,)


@app.cell
def _(wdata):
    wdata.viz_timeseries(savename="weather_timeseries_" + wdata.location)
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
    mixed_locations = ["Stovepipe", "Socorro", "Riley", "Yuma", "Utqiagvik"]
    mixed_locations = ["Socorro", "Stovepipe"]
    mixed_locations = ["Socorro", "Stovepipe", "Riley", "Utqiagvik"]

    dropdown = mo.ui.dropdown(
        options=["Yuma", "Riley", "Stovepipe", "Mercury", "Socorro", "Utqiagvik", "mix"], 
        value="mix", label="choose location"
    )
    dropdown
    return dropdown, mixed_locations


@app.cell(hide_code=True)
def _(mo):
    dropdown_time = mo.ui.dropdown(
        options=["summer", "all_yr", "winter"], 
        value="summer", label="choose season"
    )
    dropdown_time
    return (dropdown_time,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    for summer:
    - Stovepipe: opt step at 7%
    - Yuma: opt step at 12.8%
    - Socorro: opt step at 15%
    - Riley: opt step at 31%
    - Utqiagvik: opt step at 84%
    - Riley and Stovepipe: double step.
    """)
    return


@app.cell
def _():
    too_many_missing = [
        ["Stovepipe", 3, 2021],
        ["Stovepipe", 4, 2021],
        ["Stovepipe", 5, 2021],
        ["Utqiagvik", 5, 2021],
        ["Socorro", 6, 2022],
        ["Socorro", 2, 2023]
    ]
    return (too_many_missing,)


@app.cell
def _(WeatherData, np, too_many_missing):
    def get_weather_datas(locations, months, years, randomize_location=True):
        weather_datas = []
        n_avoid = 0
        n_tot = 0
        for yr in years:
            for mo in months:
                locs_to_use = [np.random.choice(locations)] if randomize_location else locations
                for location in locs_to_use:
                    n_tot += 1
                    if [location, mo, yr] in too_many_missing:
                        n_avoid += 1
                        print("SKIPPING: too much missing.")
                        continue
                    wdata = WeatherData(location, [mo], yr)
                    if wdata.all_consecutive_days():
                        weather_datas.append(wdata)
                    else:
                        n_avoid += 1
        print(f"left out: {n_avoid}/{n_tot}")
        return weather_datas

    return (get_weather_datas,)


@app.cell
def _(Weather, dropdown, dropdown_time, get_weather_datas, mixed_locations):
    season_to_months = {
        "all_yr": list(range(1, 13)),
        "summer": [5, 6, 7, 8, 9],
        "summer_met": [6, 7, 8], # meterological
        "winter": [12, 1, 2]
    }
    yrs = [2019, 2020, 2021, 2022, 2023, 2024, 2025]

    def build_weather(months):
        weather_datas = []
        if not dropdown.value == "mix":
            weather_datas = get_weather_datas([dropdown.value], months, yrs)
        elif dropdown.value == "mix":
            weather_datas = get_weather_datas(mixed_locations, months, yrs, randomize_location=False)

        return Weather(
            # list of weather data
            weather_datas,
            # tag
            dropdown.value + "_" + dropdown_time.value
        )

    weather = build_weather(season_to_months[dropdown_time.value])
    weather.ads_des_conditions
    return (weather,)


@app.cell
def _(os, weather):
    os.makedirs(weather.tag, exist_ok=True)
    return


@app.cell
def _(city_to_desert, dropdown_time, weather):
    print(dropdown_time.value)
    for wmetric in ["ads T [°C]", "des T [°C]", "ads P/P0", "des P/P0"]:
        print(wmetric)
        for _loc, _group in weather.ads_des_conditions.groupby("location"):

            print("\t" + city_to_desert[_loc])
            # print("\t\tmin = ", _group[wmetric].min())
            # print("\t\tmax = ", _group[wmetric].max())
            print("\t\tmean = ", _group[wmetric].mean())
            print("\t\tstd = ", _group[wmetric].std())
        
    for _loc, _group in weather.ads_des_conditions.groupby("location"):
        print(city_to_desert[_loc])
        print("\tmean delta p/p0: ", (_group["ads P/P0"] - _group["des P/P0"]).mean())
        print("\tmean p/p0 midpoint: ", ((_group["ads P/P0"] + _group["des P/P0"])/2).mean())
    return


@app.cell
def _(weather):
    weather.ads_des_conditions.groupby("location")["ads T [°C]"].mean().sort_values()
    return


@app.cell
def _(weather):
    weather.ads_des_conditions.groupby("location")["des T [°C]"].mean().sort_values()
    return


@app.cell
def _(weather):
    weather.ads_des_conditions.groupby("location")["ads P/P0"].mean().sort_values()
    return


@app.cell
def _(weather):
    weather.ads_des_conditions.groupby("location")["des P/P0"].mean().sort_values()
    return


@app.cell
def _(weather):
    weather.ads_des_conditions
    return


@app.cell
def _(
    T_range,
    T_ticks,
    city_to_desert,
    dropdown,
    idea_to_color,
    mixed_locations,
    p_ovr_p0_ticks,
    plt,
    sns,
    weather,
):
    with sns.plotting_context("notebook", font_scale=1.25):
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

        # Move the legend into the unused upper-right panel
        gs = pp.axes[-1, 0].get_gridspec()
        legend_ax = pp.fig.add_subplot(gs[2, 3])
        legend_ax.axis("off")

        handles = pp._legend_data.values()
        labels = pp._legend_data.keys()
        if pp.legend is not None:
            pp.legend.remove()


        labels = [city_to_desert.get(l, l) + " Desert" for l in pp._legend_data.keys()]

        legend_ax.legend(
            handles, labels, title="", loc="center", frameon=False
        )

        pp.fig.subplots_adjust(wspace=0.1, hspace=0.1)
        for c in range(4):
            pp.axes[-1, c].xaxis.labelpad = 5

        plt.savefig(weather.tag + "/ads_des_conditions.pdf", format="pdf")
        plt.show()
    return set_weather_cols_axis, short_to_proper_weather_cols, weather_cols


@app.cell
def _(weather):
    weather.tag
    return


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
def _():
    w_max = 0.4
    return (w_max,)


@app.cell
def _(mpl):
    temp_colormap = mpl.colormaps['inferno'] # or 'plasma', 'coolwarm', etc.
    return (temp_colormap,)


@app.cell
def _(BernPolyBasis, colors, inset_axes, np, plt, temp_colormap, w_max):
    class WaterAdsorptionIsotherm:
        def __init__(
            self, n, Tref=25.0, w_max=w_max, bs=None
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

        def draw(self, boundary_color=None, savename=None):
            p_over_p0s = np.linspace(0, 1.0, 100)

            fig, ax = plt.subplots()
            if boundary_color:
                fig.patch.set_edgecolor(boundary_color)
                fig.patch.set_linewidth(4)

            plt.xlabel("relative humidity, $p / [p_0(T)]$")
            plt.ylabel("water adsorption\n[kg H$_2$O/kg sorbent]")

            norm = colors.Normalize(vmin=0.0, vmax=70.0)

            for T in np.linspace(0, 70, 6):
                plt.plot(
                    p_over_p0s, 
                    [self.water_ads(T, p_over_p0) for p_over_p0 in p_over_p0s],
                    color=temp_colormap(norm(T)),
                    clip_on=False
                )

            sm = plt.cm.ScalarMappable(cmap=temp_colormap, norm=norm)
            cax = inset_axes(
                ax, width="4%", height="60%", loc="lower right",
                bbox_to_anchor=(-0.05, 0.05, 0.9, 0.95),  # (x0, y0, width, height) in axes fraction
                bbox_transform=ax.transAxes, borderpad=0
            )
            cbar = fig.colorbar(sm, cax=cax)
            cbar.set_label('temperature [°C]', labelpad=8)
            cbar.set_ticks(20*np.arange(4))

            ax.set_xlim(0, 1.0)
            ax.set_ylim(0, self.w_max)

            plt.tight_layout()
            if savename:
                plt.savefig(savename + ".pdf", format="pdf")

            plt.show()

    return (WaterAdsorptionIsotherm,)


@app.cell
def _(WaterAdsorptionIsotherm):
    wai = WaterAdsorptionIsotherm(10)
    wai.endow_stepped_isotherm(3)
    # wai.draw(boundary_color="green")
    return (wai,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 🥇 score fitness of dist'n of water deliveries
    """)
    return


@app.function
def attach_water_delivery(wai, weather, prefix=""):
    # compute water delivery
    weather.ads_des_conditions[prefix + "water del [kg H$_2$O/kg MOF]"] = wai.water_del(
        weather.ads_des_conditions
    )


@app.cell
def _():
    n_day_period = 10
    return (n_day_period,)


@app.cell
def _():
    return


@app.cell
def _(n_day_period, np, pd):
    def get_nday_totals(wai, weather, n_day_period=n_day_period, n_samples=5, seed=1337):
        attach_water_delivery(wai, weather)
        rng = np.random.default_rng(seed)
        col = "water del [kg H$_2$O/kg MOF]"

        df = weather.ads_des_conditions
        df["day"] = df["day"].astype(int)   # ensure proper int dtype once, up front

        results = []
        for (loc, month), group in df.groupby(["location", "month_period"], sort=False):
            n_days = month.days_in_month
            max_start = n_days - n_day_period + 1

            day_vals = np.zeros(n_days + 1)
            np.add.at(day_vals, group["day"].values, group[col].values)
            cum = np.cumsum(day_vals)

            starts = rng.integers(1, max_start + 1, size=n_samples)
            ends = starts + n_day_period - 1
            window_totals = cum[ends] - cum[starts - 1]

            for i in range(n_samples):
                results.append((loc, f"{month}-{i}", starts[i], ends[i], window_totals[i]))

        totals = pd.DataFrame(
            results, columns=["location", "period_label", "start_day", "end_day", "cum water del [kg/kg]"]
        ).set_index(["location", "period_label"])["cum water del [kg/kg]"]

        return totals

    return (get_nday_totals,)


@app.cell
def _(get_nday_totals, wai, weather):
    totals = get_nday_totals(wai, weather, 10, seed=3)
    totals
    return


@app.cell
def _(np):
    def var_cvar(scores, alpha):
        val_at_risk = np.percentile(scores, alpha)
        cval_at_risk = np.mean(scores[scores <= val_at_risk])
        return val_at_risk, cval_at_risk

    return (var_cvar,)


@app.cell
def _(alpha, get_nday_totals, n_day_period, var_cvar):
    def score_fitness(wai, weather, alpha=alpha, verbose=False):
        period_totals = get_nday_totals(wai, weather, n_day_period)

        per_location_var = {}
        per_location_cvar = {}
        for loc, group in period_totals.groupby("location"):
            val_at_risk, cval_at_risk = var_cvar(group.values, alpha)
            if verbose:
                print(loc)
                print("\tvar: ", val_at_risk)
                print("\tcvar: ", cval_at_risk)
            per_location_var[loc] = val_at_risk
            per_location_cvar[loc] = cval_at_risk

        min_cvar = min(per_location_cvar.values())
        if verbose:
            print("min CVaR: ", min_cvar)

        return period_totals, per_location_var, per_location_cvar, min_cvar

    return (score_fitness,)


@app.cell
def _():
    alpha = 20.0
    return (alpha,)


@app.cell
def _(score_fitness, wai, weather):
    period_totals, per_location_var, per_location_cvar, min_cvar = score_fitness(wai, weather, verbose=True)
    period_totals
    return


@app.cell
def _(
    alpha,
    city_to_desert,
    gaussian_kde,
    idea_to_color,
    n_day_period,
    np,
    plt,
    score_fitness,
    w_max,
):
    def draw_fitness_scores(wai, weather):
        max_score = w_max * n_day_period
        x_grid = np.linspace(0, max_score, 150)

        fig, ax = plt.subplots()

        loc = weather.tag.split("_")[0]
        color = idea_to_color[loc]

        period_totals, per_location_var, per_location_cvar, min_cvar = score_fitness(wai, weather)
        print("fitness [kg/kg]: ", min_cvar)
        assert period_totals.max() < max_score

        # KDE
        kde = gaussian_kde(period_totals.values)
        density = kde(x_grid)

        plt.plot(x_grid, density, color=color, lw=3, label=city_to_desert[loc] + " Desert")

        below_var = x_grid < per_location_var[loc]
        plt.fill_between(x_grid[below_var], density[below_var], alpha=0.25, color=color)

        ax.axvline(min_cvar, linestyle="--", color=color)
        ax.text(
            min_cvar, ax.get_ylim()[1] * 0.8, f"{int(100-alpha)}%-CVaR",
            ha="center", va="bottom",
            bbox=dict(facecolor="white", edgecolor="none", boxstyle="round,pad=0.2", alpha=1.0)
        )

        ax.set_xlabel("cumulative water delivery [kg H$_2$O/kg sorbent]")
        ax.set_ylabel(f"density")
        plt.xlim([0, max_score])
        plt.ylim(ymin=0)
        plt.yticks([0])
        plt.legend()
        plt.tight_layout()

        plt.savefig(weather.tag + "/best_wai_water_del_distn.pdf", format="pdf")
        plt.show()

    return (draw_fitness_scores,)


@app.cell
def _(draw_fitness_scores, wai, weather):
    if not weather.tag == "mix_summer":
        draw_fitness_scores(wai, weather)
    return


@app.cell
def _(
    alpha,
    calendar,
    city_to_desert,
    dropdown,
    idea_to_color,
    n_day_period,
    plt,
    score_fitness,
    sns,
    w_max,
):
    def viz_monthly_water_del(
        wai, weather, 
        legend_outside=dropdown.value=="mix", savename=None, boundary_color=None,
        loc_legend_loc="upper left", cvar_legend_loc="upper left", incl_cvar_legend=True, incl_loc_legend=True
    ):
        period_totals, per_location_var, per_location_cvar, min_cvar = score_fitness(wai, weather, verbose=True)

        # rename for seaborn
        period_totals = period_totals.reset_index().rename(
            columns = {
                "cum water del [kg/kg]": "cumulative water delivery\n[kg H$_2$O/kg sorbent]"
            }
        )
        period_totals["month"] = period_totals["period_label"].str.split("-").str[1].astype(int)

        fig, ax = plt.subplots()

        if boundary_color:
            fig.patch.set_edgecolor(boundary_color)
            fig.patch.set_linewidth(4)

        ax = sns.swarmplot(
            data=period_totals, 
            y="cumulative water delivery\n[kg H$_2$O/kg sorbent]", 
            x="month", 
            palette=idea_to_color,
            hue="location",
            clip_on=False,
            legend=incl_loc_legend
            # size=10
        )
        # plt.title(f"{alpha:.0f}%-CVaR: {min_cvar:.1f} kg H$_2$O/kg sorbent")

        mos = period_totals["month"].unique()
        plt.xticks(range(len(mos)), [calendar.month_abbr[mo] for mo in mos], fontsize=14)

        cvar_line = plt.axhline(min_cvar, color=idea_to_color["fitness"], linestyle="--", zorder=0, label="CVaR")
        # var_line = plt.axhline(var, color="gray", linestyle="--", zorder=0, label="VaR")
        plt.ylim([0, w_max*n_day_period])

        # Grab the location legend's handles/labels before they get overwritten
        location_handles, location_labels = ax.get_legend_handles_labels()
        # The last handle/label is the axhline ("CVaR"); split it off
        location_handles, location_labels = location_handles[:-1], location_labels[:-1]
        location_labels = [city_to_desert.get(lab, lab) for lab in location_labels]

        if incl_loc_legend:
            location_legend = ax.legend(
                location_handles, location_labels,
                title="Desert",
                bbox_to_anchor=(1.02, 1) if legend_outside else None,
                loc=loc_legend_loc,
                borderaxespad=0
            )
            ax.add_artist(location_legend)  # keep this legend from being overwritten

        if incl_cvar_legend:
            ax.legend(
                [cvar_line], [f"{100-alpha:.0f}%-CVaR"],
                bbox_to_anchor=(1.02, 0.5) if legend_outside else None,
                loc=cvar_legend_loc,
                title="fitness metric",
                borderaxespad=0
            )

        if savename:
            plt.savefig(savename + ".pdf", format="pdf", bbox_inches='tight')

        plt.show()

    # viz_monthly_water_del(wai, weather, legend_outside=False)
    return (viz_monthly_water_del,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 🎲 random WAIs to explore
    """)
    return


@app.cell
def _(
    matplotlib,
    my_colors,
    n_day_period,
    np,
    p_ovr_p0_ticks,
    plt,
    score_fitness,
    w_max,
):
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
            period_totals, per_location_var, per_location_cvar, fitness = score_fitness(wai, weather)
            print(f"fitness WAI {wai.label}: {fitness}")

            max_score = w_max * n_day_period
            bins = np.linspace(0, max_score, 17)
            assert np.max(period_totals.values) < max_score

            face_rgba = matplotlib.colors.to_rgba(the_colors[w], alpha=0.5)
            edge_rgba = matplotlib.colors.to_rgba(the_colors[w], alpha=1.0)

            ax_top.hist(
                period_totals.values, bins=bins, histtype="stepfilled",
                facecolor=face_rgba, edgecolor=edge_rgba, linewidth=1.5
            )

            ax_top.set_xlabel("cumulative water delivered\n[kg H$_2$O/kg sorbent]")
            ax_top.set_ylabel(f"# {n_day_period}-day periods")   
            ax_top.set_xlim([0.0, max_score])
            ax_top.set_ylim(ymin=0.0)

            ax_top.axvline(fitness, linestyle="--", color=the_colors[w])

        plt.show()

    return (compare_wais,)


@app.cell
def _(WaterAdsorptionIsotherm, compare_wais, np, score_fitness, weather):
    _wais = [WaterAdsorptionIsotherm(10) for i in range(51)]
    for _wai in _wais:
        _wai.endow_random_isotherm()

    _fitness = [score_fitness(wai, weather)[-1] for wai in _wais]
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
        wais, savename=None, material_labels=None, 
        the_colors=[my_colors[0]] + my_colors[3:], 
        the_linestyles=['-' for i in range(10)]
    ):
        if material_labels is None:
            material_labels = [f"#{w}" for w in range(len(wais))]

        p_over_p0s = np.linspace(0, 1.0, 100)

        fig = plt.figure()
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
                label=material_labels[w],
                lw=3, clip_on=False,
                linestyle=the_linestyles[w]
            )

        plt.xlim(0, 1.0)
        plt.ylim(0, wais[0].w_max)
        plt.legend(title="model material", fontsize=12, title_fontsize=12)
        if savename is not None:
            plt.savefig(
                savename + ".pdf", format="pdf",  bbox_inches="tight"
            )
        plt.show()

    return (viz_wais,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    random birth
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
        # if np.random.rand() < 0.0:
        wai.bs[1:-1] += delta_b
        wai.bs = np.sort(wai.bs)
        # else:
        #     wai.bs[1:-1] += np.sort(delta_b)

        wai.bs[wai.bs < 0.0] = 0.0
        wai.bs[wai.bs > wai.w_max] = wai.w_max
        wai.bs[-1] = wai.w_max

    return (mutate,)


@app.cell
def _(WaterAdsorptionIsotherm, mutate, viz_wais):
    _wais = [WaterAdsorptionIsotherm(20)]
    _wais[0].endow_random_isotherm()
    _wais.append(_wais[0].copy())
    mutate(_wais[1], 0.1)
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

        fitness = score_fitness(wai, weather)[-1]
        if verbose:
            print("---local search---")
            print("current fitness: ", fitness)

        # max out capacity at high p/p0 until fitness decreases
        for i in range(1, wai.n): # walk backwards thru array
            new_wai.bs[-i:] = wai.w_max
            new_fitness = score_fitness(new_wai, weather)[-1]
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
            new_fitness = score_fitness(new_wai, weather)[-1]
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
        wais, weather, 
        # fractions 
        f_elite=0.1, f_tourney=0.3, 
        f_rand=0.25, n_mutate=0.15, eps=0.05, verbose=False, seed=1993,
        stepify_prob=0.2
    ):  
        # what's the population size?
        pop_size = np.shape(wais)[0]

        # calculate number of elite etc
        n_elite = int(pop_size * f_elite)
        tourney_size = int(pop_size * f_tourney)
        n_rand = int(pop_size * f_rand)
        n_mutate = int(pop_size * n_mutate)

        # max water adsorption
        w_max = wais[0].w_max

        # dimension of search space
        dim = wais[0].n

        # compute fitnesses of each individual
        fitnesses = np.array([score_fitness(wai, weather)[-1] for wai in wais])

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
            if np.random.rand() < stepify_prob:
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

    fitnesses = np.array([score_fitness(wai, weather)[-1] for wai in wais])

    # second generation
    new_wais = evolve(wais, weather)
    new_fitnesses = np.array(
        [score_fitness(new_wai, weather)[-1] for new_wai in new_wais]
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
def _(evolve, gen_initial_pop, ls_stepify, np, score_fitness):
    def do_evolution(weather, n_generations, pop_size, dim, stepify_prob=0.2, seed=137):
        np.random.seed(seed)

        # generate population
        wais = gen_initial_pop(pop_size, dim)

        # score fitnesses
        fitnesses = np.array([score_fitness(wai, weather)[-1] for wai in wais])

        # store progress
        fitnesses_gen = [fitnesses]
        best_wai_gen = [wais[np.argmax(fitnesses)]]

        # evolve over generations
        for g in range(1, n_generations):
            print("Gen: ", g)
            wais = evolve(wais, weather, stepify_prob=stepify_prob)
            fitnesses = np.array([score_fitness(wai, weather)[-1] for wai in wais])

            fitnesses_gen.append(fitnesses)
            best_wai_gen.append(wais[np.argmax(fitnesses)])

        best_wai = wais[np.argmax(fitnesses)]
        ls_stepify(best_wai, weather)
        best_wai.label = "optimal"
        best_period_totals, _, _, best_fitness = score_fitness(best_wai, weather, verbose=True)

        return fitnesses_gen, best_wai_gen, best_wai, best_period_totals, best_fitness

    return (do_evolution,)


@app.cell
def _(do_evolution, run_evol_cbox, weather):
    is_toy_scenario = "Riley" in weather.tag

    if is_toy_scenario:
        print("TOY SCENARIO")
        pop_size = 10
        n_generations = 20
    else:
        pop_size = 30
        n_generations = 30
    n = 50
    if run_evol_cbox.value:
        fitnesses_gen, best_wai_gen, best_wai, best_period_totals, best_fitness = do_evolution(
            weather, n_generations, pop_size, n, stepify_prob=0.05 if is_toy_scenario else 0.2, seed=12
        )
    return best_fitness, best_wai, best_wai_gen, fitnesses_gen, n


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### viz opt isotherm and its fitness
    """)
    return


@app.cell
def _(best_wai, dropdown, idea_to_color, weather):
    best_wai.draw(
        boundary_color=idea_to_color["mix"] if "mix" in dropdown.value else None,
        savename=weather.tag + f"/best_wai"
    )
    return


@app.cell
def _(best_wai):
    best_wai.get_p_ovr_p0_half_max()
    return


@app.cell
def _(best_wai, draw_fitness_scores, weather):
    draw_fitness_scores(best_wai, weather)
    return


@app.cell
def _(best_wai, viz_monthly_water_del, weather):
    viz_monthly_water_del(
        best_wai, weather,
        # boundary_color=None if dropdown.value == "mix" else idea_to_color[dropdown.value],
        savename=weather.tag + f"/best_wai_fitness",
        legend_outside=False,
        cvar_legend_loc="upper left", loc_legend_loc="upper right", incl_cvar_legend=True
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### analyze progress
    """)
    return


@app.cell
def _(alpha, dropdown, fitnesses_gen, np, pd, plt, sns, weather):
    def viz_fitness_progress(fitnesses_gen):
        data = pd.DataFrame(
            [
                [g, fitness] for g, fitnesses in enumerate(fitnesses_gen) 
                for fitness in fitnesses
            ]
            ,
            columns=['generation', 'fitness [kg H$_2$O/kg sorbent]']
        )

        fig, ax = plt.subplots()

        sns.stripplot(
            data, 
            x="generation", y="fitness [kg H$_2$O/kg sorbent]",
            hue="generation", color="C2", palette="crest", legend=False,
            ax=ax, clip_on=False
        )
        plt.ylabel(f"{100-alpha:.0f}%-CVaR\n cumulative water delivery\n[kg H$_2$O/kg sorbent]")
        plt.tick_params(axis='x', labelrotation=90)
        # plt.axhline(
        #     y=step_fitnesses[id_opt_step], 
        #     color="gray", linestyle="--", zorder=-1
        # )
        if not dropdown.value == "Riley":
            plt.xticks(2*np.arange(int(data["generation"].max()/2)+1))
        plt.ylim(ymin=0)
        plt.tight_layout()
        plt.savefig(
             weather.tag + "/fitness_progress.pdf", format="pdf"
        )
        plt.show()

    viz_fitness_progress(fitnesses_gen)
    return


@app.cell
def _(dropdown):
    dropdown.value
    return


@app.cell
def _(
    best_wai_gen,
    colors,
    mpl,
    np,
    p_ovr_p0_ticks,
    plt,
    w_max,
    wais,
    weather,
):
    def viz_best_wais(best_wai_gen):
        p_over_p0s = np.linspace(0, 1.0, 150)
        Tref = best_wai_gen[0].Tref

        colormap = mpl.colormaps['crest'] # or 'plasma', 'coolwarm', etc.
        norm = colors.Normalize(vmin=0, vmax=len(best_wai_gen))

        plt.figure()
        plt.xlabel("relative humidity, $p/p_0[T]$")
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
            [0.7, 0.15, 0.2, 0.6]
        )
        cb_ax.axis("off")
        plt.colorbar(
            sm, ax=cb_ax, label='generation', 
        )
        plt.xlim([0, 1])
        plt.ylim([0, w_max])

        plt.tight_layout()
        plt.savefig(
            weather.tag + "/wai_progress.pdf", format="pdf"
        )
        plt.show()

    viz_best_wais(best_wai_gen)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### pickle for use later
    """)
    return


@app.cell
def _(os):
    os.makedirs("pkls", exist_ok=True)
    return


@app.cell
def _(pickle):
    def pickle_this(var, pf_name):
        pf_name = "pkls/" + pf_name + '.pkl'

        with open(pf_name, 'wb') as pf:
            pickle.dump(var, pf)
            print("saved in: ", pf_name)

    return (pickle_this,)


@app.cell
def _(best_wai, pickle_this, weather):
    attach_water_delivery(best_wai, weather)
    pickle_this(weather, weather.tag + "_weather")
    pickle_this(best_wai, weather.tag + "_opt_isotherm")
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
    set_weather_cols_axis,
    short_to_proper_weather_cols,
    sns,
    weather,
    weather_cols,
):
    def viz_daily_performance(best_wai, weather):
        attach_water_delivery(best_wai, weather)
        performance_data = weather.ads_des_conditions.copy()
        performance_data = performance_data.rename(
            columns={"water del [kg H$_2$O/kg MOF]": "water delivery\n[kg H$_2$O/kg MOF]"}
        )
        performance_data = performance_data.sample(frac=1, random_state=42).reset_index(drop=True) # shuffle

        cols_to_plot = weather_cols + ["water delivery\n[kg H$_2$O/kg MOF]"]

        # Initialize the grid
        pp = sns.PairGrid(
            performance_data.rename(
                columns=short_to_proper_weather_cols
            ),
            vars=[short_to_proper_weather_cols[w] for w in weather_cols] + ["water delivery\n[kg H$_2$O/kg MOF]"],
            hue="water delivery\n[kg H$_2$O/kg MOF]", 
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
            weather.tag + "/daily_performance.pdf", format="pdf",
            bbox_inches="tight"
        )

        plt.show()

    viz_daily_performance(best_wai, weather)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### viz water delivery on a day
    """)
    return


@app.cell
def _(T_range, colors, np, p_ovr_p0_ticks, plt, temp_colormap):
    def viz_water_del(wai, weather, date, savename=None, boundary_color=None):
        day_data = weather.ads_des_conditions[
            weather.ads_des_conditions["date"].apply(
                lambda d: d == date
            )
        ].iloc[0, :]

        p_over_p0s = np.linspace(0, 1.0, 100)

        fig, ax = plt.subplots()
        if boundary_color:
            fig.patch.set_edgecolor(boundary_color)
            fig.patch.set_linewidth(4)
        plt.xlabel("relative humidity, $p / [p_0(T)]$")
        plt.xticks(p_ovr_p0_ticks)
        plt.xlim(0, 1.0)
        plt.ylabel("water adsorption\n[kg H$_2$O/kg sorbent]")

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
                color=temp_colormap(norm(T)),
                label=f"T = {T:0.1f}°C",
                lw=3, clip_on=False
            )
            plt.scatter(
                p_ovr_p0, w,
                color=temp_colormap(norm(T)), label=label, zorder=25,
                marker="*", 
                edgecolor="black",
                s=150, clip_on=False
            )

        # put water delivery there
        plt.arrow(
            p_ovr_p0_night, w_night, 0, w_day - w_night, 
            color="black", head_width=0.008, length_includes_head=True,
            lw=2, zorder=10
        )
        wdel_label = f" water delivery:\n {w_night-w_day:0.2f} kg/kg"

        # plt.text(
        #     p_ovr_p0_night + 0.01, (w_day + w_night) / 2,
        #     wdel_label,
        #     color='black', 
        #     fontsize=10, 
        #     verticalalignment='center'
        # )
        plt.plot(
            [p_ovr_p0_night, p_ovr_p0_day], [w_day, w_day], 
            color="gray", linestyle="--"
        )

        plt.legend(fontsize=12, title=str(date) + "\n" + wdel_label)
        plt.xlim([0, 1])
        plt.ylim(ymin=0.0)

        if savename:
            plt.savefig(
                savename + ".pdf", format="pdf", bbox_inches="tight"
            )
        plt.show()

    return (viz_water_del,)


@app.cell
def _(weather):
    weather.ads_des_conditions["date"]
    return


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
    viz_water_del(
        best_wai, weather, failure_list.iloc[failure_explorer.value]["date"],
        savename=weather.tag + "/typical_failure"
    )
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
            [score_fitness(wai, weather)[-1] for wai in wais]
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
def _(best_wai, best_wai_step, idea_to_color, viz_wais, weather):
    viz_wais(
        [best_wai, best_wai_step],
        material_labels=[f"optimal", f"optimal sharp-S"], 
        the_colors=[idea_to_color[weather.tag.split("_")[0]], idea_to_color["step"]],
        savename=weather.tag + f"/opt_Vs_opt_S"
    )
    return


@app.cell
def _(best_wai_step):
    best_wai_step.get_p_ovr_p0_half_max(verbose=True)
    return


@app.cell
def _(best_wai_step, viz_monthly_water_del, weather):
    viz_monthly_water_del(
        best_wai_step, weather, 
        incl_cvar_legend=False, 
        incl_loc_legend=False,
        savename=weather.tag + f"/best_step_wai_fitness",
    )
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
            weather.tag + "/step_search.pdf",
            format="pdf",  bbox_inches="tight"
        )

        plt.show()

    viz_step_wais(step_wais, step_fitnesses, id_opt_step)
    return


@app.cell
def _(my_colors):
    my_colors
    return


@app.cell
def _(
    figsize,
    gaussian_kde,
    idea_to_color,
    n_day_period,
    np,
    plt,
    score_fitness,
    w_max,
):
    def compare_best_wai_and_best_wai_step(best_wai, best_wai_step, weather):
        max_score = w_max * n_day_period
        x_grid = np.linspace(0, max_score, 150)

        fig, ax = plt.subplots(figsize=[figsize[0], figsize[1]*0.6])

        the_colors = [idea_to_color[weather.tag.split("_")[0]], idea_to_color["step"]]
        labels = ["optimal", "optimal sharp-S"]
        for w, wai in enumerate([best_wai, best_wai_step]):
            period_totals, per_location_var, per_location_cvar, min_cvar = score_fitness(wai, weather)
            print("\tfitness [kg/kg]: ", min_cvar)
            print("\tmean [kg/kg]: ", period_totals.mean())
            assert period_totals.max() < max_score

            # KDE
            kde = gaussian_kde(period_totals.values)
            density = kde(x_grid)

            plt.plot(x_grid, density, color=the_colors[w], lw=3, label=labels[w], clip_on=False)
            # below_var = x_grid < np.mean(list(per_location_var.values()))
            # plt.fill_between(x_grid[below_var], density[below_var], alpha=0.25, color=the_colors[w])

            ax.axvline(min_cvar, linestyle="--", color=the_colors[w], clip_on=False)

        ax.set_xlabel("cumulative water delivery [kg H$_2$O/kg sorbent]")
        ax.set_ylabel(f"density")
        plt.xlim([0, max_score])
        plt.ylim(ymin=0)
        plt.yticks([0])
        plt.legend()
        plt.tight_layout()

        plt.savefig(weather.tag + "/comparison_w_step.pdf", format="pdf")
        plt.show()

    return (compare_best_wai_and_best_wai_step,)


@app.cell
def _(best_wai, best_wai_step, compare_best_wai_and_best_wai_step, weather):
    compare_best_wai_and_best_wai_step(best_wai, best_wai_step, weather)
    return


@app.cell
def _(best_fitness, best_fitness_step):
    print(
        "% mass savings over a step: ",
        (best_fitness - best_fitness_step) / best_fitness_step * 100
    )
    return


@app.cell
def _(best_wai_step, dropdown, idea_to_color, weather):
    best_wai_step.draw(
        boundary_color=idea_to_color["step"] if "mix" in dropdown.value else None,
        savename=weather.tag + "/opt_step"
    )
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # compare single-location tailored optimal WAIs

    for all locs during summer.
    """)
    return


@app.cell
def _(pickle):
    def unpickle(pf_name):
        pf_name = "pkls/" + pf_name + '.pkl'
        with open(pf_name, 'rb') as pf:
            var = pickle.load(pf)
        return var

    return (unpickle,)


@app.cell
def _(os):
    os.makedirs("comparison", exist_ok=True)
    return


@app.cell
def _(mixed_locations, unpickle):
    best_wais = [unpickle(loc + "_summer_opt_isotherm") for loc in mixed_locations]
    weathers = [unpickle(loc + "_summer_weather") for loc in mixed_locations]
    return best_wais, weathers


@app.cell
def _(
    city_to_desert,
    gaussian_kde,
    idea_to_color,
    n_day_period,
    np,
    plt,
    score_fitness,
    w_max,
):
    def compare_all_wai_fitness(wais, weathers):
        max_score = w_max * n_day_period
        x_grid = np.linspace(0, max_score, 150)

        fig, ax = plt.subplots()

        for wai, weather in zip(wais, weathers):
            loc = weather.tag.split("_")[0]
            color = idea_to_color[loc]

            period_totals, per_location_var, per_location_cvar, min_cvar = score_fitness(wai, weather)
            print("fitness [kg/kg]: ", min_cvar)
            assert period_totals.max() < max_score

            # KDE
            kde = gaussian_kde(period_totals.values)
            density = kde(x_grid)

            plt.plot(x_grid, density, color=color, lw=3, label=city_to_desert[loc])

            below_var = x_grid < per_location_var[loc]
            plt.fill_between(x_grid[below_var], density[below_var], alpha=0.05, color=color)

            ax.axvline(min_cvar, linestyle="--", color=color)

        ax.set_xlabel("cumulative water delivery [kg H$_2$O/kg sorbent]")
        ax.set_ylabel(f"density")   

        ax.set_xlim([0.0, max_score])
        ax.set_ylim(ymin=0.0)

        plt.legend(title="Desert")
        plt.tight_layout()
        plt.savefig("comparison/compare_fitnesses.pdf", format="pdf")
        plt.show()

    return (compare_all_wai_fitness,)


@app.cell
def _(best_wais, compare_all_wai_fitness, weathers):
    compare_all_wai_fitness(best_wais, weathers)
    return


@app.cell
def _(city_to_desert, idea_to_color, np, p_ovr_p0_ticks, plt):
    def compare_best_wais(wais, weathers):
        p_over_p0s = np.linspace(0, 1.0, 100)

        fig = plt.figure()
        plt.xlabel("relative humidity $p / [p_0(T)]$")
        plt.xticks(p_ovr_p0_ticks)
        plt.ylabel(
            f"water adsorption at {wais[0].Tref:.0f}°C\n[kg H$_2$O/kg sorbent]"
        )

        for wai, weather in zip(wais, weathers):
            loc = weather.tag.split("_")[0]
            print(loc)
            print(wai.get_p_ovr_p0_half_max())
            plt.plot(
                p_over_p0s, 
                [wai.water_ads(wai.Tref, p_over_p0) for p_over_p0 in p_over_p0s],
                color=idea_to_color[loc],
                label=city_to_desert[loc],
                lw=3,
                clip_on=False
            )

        plt.xlim(0, 1.0)
        plt.ylim(0, wais[0].w_max)
        plt.legend(title="Desert")
        plt.savefig(
            "comparison/best_wai_comparison.pdf", format="pdf", bbox_inches="tight"
        )
        plt.show()

    return (compare_best_wais,)


@app.cell
def _(best_wais, compare_best_wais, weathers):
    compare_best_wais(best_wais, weathers)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # how does the opt isotherm from one city transfer to another?
    """)
    return


@app.cell
def _(unpickle):
    # bring A -> B
    loc_A = "Stovepipe"
    season_A = "summer"

    loc_B = "Riley"
    season_B = "summer"

    comparison_case = f"wai_for_{loc_A}_{season_A}_operating_in_{loc_B}_{season_B}"

    best_wai_A = unpickle(f"{loc_A}_{season_A}_opt_isotherm")

    best_wai_B = unpickle(f"{loc_B}_{season_B}_opt_isotherm")
    weather_B = unpickle(f"{loc_B}_{season_B}_weather")
    return (
        best_wai_A,
        best_wai_B,
        comparison_case,
        loc_A,
        loc_B,
        season_A,
        season_B,
        weather_B,
    )


@app.cell
def _(
    best_wai_A,
    best_wai_B,
    comparison_case,
    loc_A,
    loc_B,
    season_A,
    season_B,
    viz_wais,
):
    viz_wais(
        [best_wai_A, best_wai_B],
        material_labels=[f"opt for {season_A} in {loc_A}", f"opt for {season_B} in {loc_B}"], 
        savename=f"comparison/both_wais_{comparison_case}"
    )
    return


@app.cell
def _(viz_cities):
    viz_cities(
        ["Riley"], 
        savename=f"map_Riley",
        extent=[-127, -110, 30, 50],
        xy_shift = {
            "Riley": [3.5, 0]
        }
    )
    return


@app.cell
def _(viz_cities):
    viz_cities(
        ["Stovepipe"], 
        savename=f"map_Stovepipe",
        extent=[-127, -110, 30, 50],
        xy_shift = {
            "Stovepipe": [3.5, 0]
        }
    )
    return


@app.cell
def _(comparison_case, loc_A, loc_B, viz_cities):
    viz_cities(
        [loc_A, loc_B], 
        city_AB=[loc_A, loc_B], 
        savename=f"comparison/map_{comparison_case}",
        extent=[-127, -102, 30, 50],
        xy_shift = {
            "Riley": [3.5, 0],
            "Stovepipe": [-3, 0.0],
            "Socorro": [0, 2.5],
            "Utqiagvik": [0, -3.75],
            "Yuma": [-7.5, 0]
        }
    )
    return


@app.cell
def _(
    alpha,
    city_to_desert,
    comparison_case,
    gaussian_kde,
    idea_to_color,
    my_colors,
    n_day_period,
    np,
    plt,
    score_fitness,
    w_max,
):
    def viz_mismatch_fitness(wai, weather, loc_A, loc_B):
        max_score = w_max * n_day_period
        x_grid = np.linspace(0, max_score, 150)

        fig, ax = plt.subplots()
        # fig.patch.set_edgecolor(idea_to_color[loc_A])
        # fig.patch.set_linewidth(4)

        color = my_colors[-1] # gray
        color = idea_to_color[loc_B]

        period_totals, per_location_var, per_location_cvar, min_cvar = score_fitness(wai, weather)
        print("fitness [kg/kg]: ", min_cvar)
        assert period_totals.max() < max_score

        # KDE
        kde = gaussian_kde(period_totals.values)
        density = kde(x_grid)

        label = f"opt sorbent for {city_to_desert[loc_A]} Desert\n  operating in {city_to_desert[loc_B]} Desert\n  {100-alpha:.0f}%-CVaR: {min_cvar:.1f} kg/kg"
        plt.plot(x_grid, density, color=color, lw=3, label=label)

        below_var = x_grid < per_location_var[loc_B]
        plt.fill_between(x_grid[below_var], density[below_var], alpha=0.25, color=color)

        ax.axvline(min_cvar, linestyle="--", color=color)
        # ax.text(
        #     min_cvar, ax.get_ylim()[1] * 0.8, f"{int(alpha)}%-CVaR",
        #     ha="center", va="bottom",
        #     bbox=dict(facecolor="white", edgecolor="none", boxstyle="round,pad=0.2", alpha=1.0)
        # )

        ax.set_xlabel("cumulative water delivery\n[kg H$_2$O/kg sorbent]")
        ax.set_ylabel(f"density")   

        ax.set_xlim([0.0, max_score])
        ax.set_ylim(ymin=0.0)

        plt.legend()
        plt.yticks([0])

        plt.tight_layout()
        plt.savefig(f"comparison/{comparison_case}_fitness.pdf", format="pdf")
        plt.show()

    return (viz_mismatch_fitness,)


@app.cell
def _(best_wai_A, loc_A, loc_B, viz_mismatch_fitness, weather_B):
    viz_mismatch_fitness(best_wai_A, weather_B, loc_A, loc_B)
    return


@app.cell
def _(best_wai_A, best_wai_B, weather_B):
    attach_water_delivery(best_wai_A, weather_B, prefix="A_")
    attach_water_delivery(best_wai_B, weather_B, prefix="B_")
    wdel_diff_data = weather_B.ads_des_conditions.copy()
    # we are in city B. we want to find when sorbent optimized for B does way better than the sorbent optimized for A.
    wdel_diff_data["water del B - A"] = wdel_diff_data["B_water del [kg H$_2$O/kg MOF]"] - wdel_diff_data["A_water del [kg H$_2$O/kg MOF]"]
    wdel_diff_data = wdel_diff_data.sort_values(by="water del B - A", ascending=False)
    wdel_diff_data
    return (wdel_diff_data,)


@app.cell
def _(best_wai_A):
    best_wai_A.get_p_ovr_p0_half_max()
    return


@app.cell
def _(best_wai_B):
    best_wai_B.get_p_ovr_p0_half_max()
    return


@app.cell
def _(best_wai_A, comparison_case, viz_water_del, wdel_diff_data, weather_B):
    viz_water_del(
        best_wai_A, weather_B, 
        wdel_diff_data.iloc[0]["date"],
        # boundary_color=idea_to_color[other_tag],
        savename=f"comparison/failure_{comparison_case}"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # compare winter and summer in Mojave
    """)
    return


@app.cell
def _(idea_to_color, unpickle, viz_wais):
    _best_wai_winter = unpickle(f"Stovepipe_winter_opt_isotherm")
    _best_wai_winter.get_p_ovr_p0_half_max(verbose=True)
    _best_wai_summer = unpickle(f"Stovepipe_summer_opt_isotherm")
    _best_wai_summer.get_p_ovr_p0_half_max(verbose=True)

    viz_wais(
        [_best_wai_summer, _best_wai_winter],
        material_labels=[f"optimal May-Sep", f"optimal Dec-Feb"], 
        the_colors=[idea_to_color["Stovepipe"], idea_to_color["Stovepipe"]],
        the_linestyles=['-', '--'],
        savename="Stovepipe_winter/opt_summer_v_winter"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # shape-matching
    """)
    return


@app.cell
def _(np, pd):
    class ExptIsotherm:
        def __init__(self, name, T):
            self.name = name
            self.T = T

            # read ads data
            filename = f"{self.name}_{self.T}C.csv"
            print(filename)
            url = f"https://github.com/SimonEnsemble/water_harvesting/raw/refs/heads/main/new/data/{filename}"
            self.data = pd.read_csv(url)
        
            if self.data["RH[%]"].max() <= 1:
                self.data["RH[%]"] = self.data["RH[%]"] * 100
            
            self.data = self.data.sort_values("RH[%]")

        def water_ads(self, p_ovr_p0):
            return np.interp(
                # RH query
                np.asarray(p_ovr_p0) * 100, 
                # RH in data
                self.data["RH[%]"].to_numpy(), 
                # water ads in data
                self.data["Water Uptake [kg kg-1]"].to_numpy()
            )

    return (ExptIsotherm,)


@app.cell
def _():
    data_I_got = [
        ["MOF-801", 25],
        ["KMF-1", 25],
        ["CAU-23", 25],
        ["MIL-160", 20],
        ["MOF-303", 25],
        ["CAU-10-H", 25],
        ["Al-Fum", 25],
        ["MIP-200", 30]
    ]
    return (data_I_got,)


@app.cell
def _(ExptIsotherm, data_I_got):
    expt_isotherms = {
        mof: ExptIsotherm(mof, T) for mof, T in data_I_got
    }
    return (expt_isotherms,)


@app.cell
def _(city_to_desert, idea_to_color, np, plt):
    def draw_shape_match(wai, expt_isotherm, savename=None, loc=None, season=""):
        T = expt_isotherm.T
    
        fig, ax = plt.subplots()

        plt.xlabel("relative humidity, $p / [p_0(T)]$")
        plt.ylabel(f"water adsorption at {T}°C\n[kg H$_2$O/kg MOF]")

        # target WAI
        color = idea_to_color[loc]
        p_ovr_p0s = np.linspace(0, 1, 250)
        ws = wai.water_ads(T, p_ovr_p0s)
        plt.plot(p_ovr_p0s, ws, label=f"optimal for\n{city_to_desert[loc]} Desert\n({season})", lw=3, color=color)

        # exp'tl isotherm
        ax.scatter(
            expt_isotherm.data["RH[%]"] / 100.0, expt_isotherm.data["Water Uptake [kg kg-1]"],
            label=f"{expt_isotherm.name}", color="black", s=40, zorder=10
        )
        p_ovr_p0s = np.linspace(0, expt_isotherm.data["RH[%]"].max()/100, 250)
        ax.plot(p_ovr_p0s, expt_isotherm.water_ads(p_ovr_p0s), color="black", lw=3, zorder=10)

        plt.xlim([0, 1])
        plt.ylim(ymin=0)

        plt.legend()
        plt.tight_layout()
        if savename:
            plt.savefig(savename + ".pdf", format="pdf", bbox_inches="tight")

        plt.show()

    return (draw_shape_match,)


@app.cell
def _(draw_shape_match, expt_isotherms, unpickle):
    draw_shape_match(
        unpickle("Riley_summer_opt_isotherm"),
        expt_isotherms["CAU-23"],
        loc="Riley",
        season="May-Sep",
        savename="Riley_summer/shape_match"
    )
    return


@app.cell
def _(np):
    def loss(x, wai, expt_isotherms):
        T = expt_isotherms[0].T
    
        p_ovr_p0s = np.linspace(0, 0.9, 35)
    
        n_mix = np.sum(
            x[i] * expt_isotherm.water_ads(p_ovr_p0s) for i, expt_isotherm in enumerate(expt_isotherms)
        )
    
        n_target = wai.water_ads(T, p_ovr_p0s)

        return np.sum((n_mix - n_target) ** 2)

    return (loss,)


@app.cell
def _(expt_isotherms, loss, wai):
    loss([0.2, 0.8], wai, [expt_isotherms["KMF-1"], expt_isotherms["MOF-801"]])
    return


@app.cell
def _(expt_isotherms):
    _mofs = ["MOF-801", "Al-Fum", "KMF-1"]
    # _mofs = [mof for mof, T in data_I_got if T == 25]
    mofs_to_mix = [expt_isotherms[mof] for mof in _mofs]
    return (mofs_to_mix,)


@app.cell
def _(loss, minimize, np):
    def do_shape_matching(wai, expt_isotherms):
        n = len(expt_isotherms)

        # initial guess: uniform weights
        x0 = np.ones(n) / n
    
        # bounds: each x_i in [0, 1]
        bounds = [(0, 1) for _ in range(n)]
    
        # constraint: sum(x) == 1
        constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
    
        result = minimize(
            loss,
            x0,
            args=(wai, expt_isotherms),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )
    
        x_opt = result.x

        for i, expt_isotherm in enumerate(expt_isotherms):
            print(f"{expt_isotherm.name}: {x_opt[i]}")

        return x_opt

    return (do_shape_matching,)


@app.cell
def _(do_shape_matching, mofs_to_mix, unpickle):
    x_opt = do_shape_matching(unpickle("Stovepipe_winter_opt_isotherm"), mofs_to_mix)
    return (x_opt,)


@app.cell
def _(city_to_desert, idea_to_color, np, plt):
    def draw_mixed_shape_match(wai, expt_isotherms, x, savename=None, loc=None, season=None):
        T = expt_isotherms[0].T
        for expt_isotherm in expt_isotherms:
            assert np.isclose(expt_isotherm.T, T)

        p_ovr_p0_max = np.min([expt_isotherm.data["RH[%]"].max()/100 for expt_isotherm in expt_isotherms])
    
        fig, ax = plt.subplots()

        plt.xlabel("relative humidity, $p / [p_0(T)]$")
        plt.ylabel(f"water adsorption at {T}°C\n[kg H$_2$O/kg MOF]")

        # target WAI
        color = idea_to_color[loc]
        p_ovr_p0s = np.linspace(0, 1, 250)
        ws = wai.water_ads(T, p_ovr_p0s)
        plt.plot(p_ovr_p0s, ws, label=f"optimal for\n{city_to_desert[loc]} Desert\n({season})", lw=3, color=color)

        # exp'tl isotherm mixed
        p_ovr_p0s = np.linspace(0, p_ovr_p0_max, 100)
        n = np.sum(
            x[i] * expt_isotherm.water_ads(p_ovr_p0s) for i, expt_isotherm in enumerate(expt_isotherms)
        )
        label = ""
        for i, expt_isotherm in enumerate(expt_isotherms):
            label += f"{x[i]*100:.0f}% {expt_isotherm.name}\n"
        label = label[:-1]
        ax.plot(p_ovr_p0s, n, color="black", lw=3, zorder=10, label=label)

        # markers = ["s", "o", "D"]
        # for i, expt_isotherm in enumerate(expt_isotherms):
        #     label = f"{expt_isotherm.name} ({x[i]*100:.0f}%)"
        #     ax.scatter(
        #         expt_isotherm.data["RH[%]"] / 100.0, x[i] * expt_isotherm.data["Water Uptake [kg kg-1]"],
        #         label=label, color="black", s=40, zorder=10, marker=markers[i]
        #     )

        plt.xlim([0, 1])
        plt.ylim(ymin=0)

        plt.legend(loc="lower right")# , bbox_to_anchor=(1.02, 1))
        plt.tight_layout()

        if savename:
            plt.savefig(savename + ".pdf", format="pdf", bbox_inches="tight")

        plt.show()

    return (draw_mixed_shape_match,)


@app.cell
def _(draw_mixed_shape_match, mofs_to_mix, unpickle, x_opt):
    draw_mixed_shape_match(
        unpickle("Stovepipe_winter_opt_isotherm"),
        mofs_to_mix,
        x_opt,
        loc="Stovepipe",
        season="Dec-Feb",
        savename="Stovepipe_winter/shape_match"
    )
    return


@app.cell
def _(data_I_got, expt_isotherms, plt, sns):
    def draw_mof_ads_data(expt_isotherms):
        T = expt_isotherms[0].T

        fig, ax = plt.subplots()

        plt.xlabel("relative humidity, $p / [p_0(T)]$")
        plt.ylabel(f"water adsorption at {T}°C\n[kg H$_2$O/kg MOF]")
    
        markers = ['o', 's', '^', 'D', 'x', '*']
        colors = sns.color_palette("pastel", len(expt_isotherms))
        for i, expt_isotherm in enumerate(expt_isotherms):
            label = f"{expt_isotherm.name}"
            ax.scatter(
                expt_isotherm.data["RH[%]"] / 100.0, expt_isotherm.data["Water Uptake [kg kg-1]"],
                label=label, color=colors[i], s=40, zorder=10, marker=markers[i]
            )

        plt.xlim([0, 1])
        plt.ylim(ymin=0)

        plt.legend(loc="lower right")# , bbox_to_anchor=(1.02, 1))
        plt.tight_layout()

        plt.savefig("expt_isotherms.pdf", format="pdf", bbox_inches="tight")

        plt.show()

    _expt_isotherms = [expt_isotherms[mof] for mof, T in data_I_got if T == 25]
    _expt_isotherms = [expt_isotherms[mof] for mof in ["MOF-801", "Al-Fum", "KMF-1", "CAU-23"]]
    draw_mof_ads_data(_expt_isotherms)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
