import marimo

__generated_with = "0.17.6"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import math
    import numpy as np
    import os
    import warnings
    import matplotlib.dates as mdates
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import matplotlib.colors as colors
    import seaborn as sns
    from aquarel import load_theme

    theme = load_theme("minimal_light")
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
        colors,
        math,
        mo,
        mpl,
        my_date_format,
        np,
        os,
        pd,
        plt,
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
    time_to_color = {'day': my_colors[1], "night": my_colors[2]}
    time_to_color["ads"] = time_to_color["night"]
    time_to_color["des"] = time_to_color["day"]
    return (time_to_color,)


@app.cell
def _(os):
    fig_dir = "figs"
    os.makedirs(fig_dir, exist_ok=True)
    return (fig_dir,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # modeling the vapor pressure of water
    """)
    return


@app.function
# input  T  : deg C
# output P* : bar
def water_vapor_presssure(T):
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
    elif T+273.15 > 255.9 and T+273.15 < 373.0: # low temp
        A = 4.6543
        B = 1435.264
        C = -64.848
    # T in [379, 573] K
    elif T+273.15 > 379.0 and T+273.15 < 573.0: # high temp
        A = 3.55959
        B = 643.748
        C = -198.043

    return 10.0 ** (A - B / ((T + 273.15) + C))


@app.cell
def _():
    water_vapor_presssure(100.0) # around 1 ATM
    return


@app.cell
def _():
    water_vapor_presssure(20.0) # 0.023 atm
    return


@app.cell
def _(np, plt):
    def viz_water_vapor_presssure():
        Ts = np.linspace(-5.0, 100.0, 250) # deg C

        plt.figure()
        plt.xlabel("T [°C]")
        plt.ylabel("P* [bar]")
        plt.plot(Ts, [water_vapor_presssure(T_i) for T_i in Ts], linewidth=3)
        plt.scatter(100.0, water_vapor_presssure(100.0))
        plt.show()

    viz_water_vapor_presssure()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # ☀️ weather time series data (for capture and release conditions)

    NOAA hourly data [here](https://www.ncei.noaa.gov/access/crn/products.html).

    no need to pre-process now.
    """)
    return


@app.cell
def _():
    city_to_state = {
        'Tucson': 'AZ', 
        'Socorro': 'NM', 
        'Utqiagvik': 'AK', 
        'Mercury': 'NV', 
        'Stovepipe': 'CA'
    }
    return (city_to_state,)


@app.cell
def _(city_to_state, fig_dir, my_date_format, np, os, pd, plt, time_to_color):
    class Weather:
        def __init__(
            self, months, year, location, time_to_hour={'day': 15, 'night': 5}
        ):
            self.months = months
            self.year = year
            self.location = location

            print(f"reading {year} {location} weather.")
            print("\tnighttime adsorption hr: ", time_to_hour["night"])
            print("\tdaytime harvest hr: ",      time_to_hour["day"])

            self.relevant_weather_cols = [
                "T_HR_AVG", "RH_HR_AVG", "SUR_TEMP", "SUR_RH_HR_AVG"
            ] # latter inferred

            self.time_to_hour = time_to_hour

            self._read_raw_weather_data()

            self._filter_missing()

            self._process_datetime_and_filter()

            self._minimalize_raw_data()

            self._day_night_data()

            self._gen_ads_des_conditions()
            self._compute_p_ovr_p0_max()
            self._compute_T_range()

            # for plots
            self.loc_title = f"{self.location}, {city_to_state[self.location]}."
            self.save_tag = fig_dir + f"/{self.location}_"

        def _read_raw_weather_data(self):
            wdata_dir = "data/"
            wfiles = os.listdir(wdata_dir)
            assert [self.location in wfile for wfile in wfiles]

            filename = list(
                filter(
                    lambda wfile: self.location in wfile and str(self.year) in wfile, 
                    wfiles
                )
            )
            assert len(filename) == 1
            filename = filename[0]
            print(f"\t...reading weather data from {filename}")

            self.raw_data = pd.read_csv(
                wdata_dir + "/" + filename,
                names=open(wdata_dir + "/headers.txt", "r").readlines()[1].split(), 
                dtype={'LST_DATE': str}, 
                sep='\s+'
            )

            self._remove_rainy_days()

        def _remove_rainy_days(self):
            print("removing rainy days")
            rain_group_by_day = self.raw_data.groupby("LST_DATE")["P_CALC"]
            print("\t# rainy days: ", (rain_group_by_day.sum() > 0.0).sum())
            ids = rain_group_by_day.transform("sum") == 0.0
            self.raw_data = self.raw_data[ids]

        def _process_datetime_and_filter(self):
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
            self.raw_data["datetime"] = self.raw_data["date"] + self.raw_data["time"]

            # filter by month
            self.raw_data = self.raw_data.loc[
                [m in self.months for m in self.raw_data["datetime"].dt.month]
            ]

            self._infer_surface_RH()

        def _infer_surface_RH(self):
            # compute new relative humidity at surface temperature, for heated air
            # partial pressure @ ambient:
            #      RH * p0(T)
            #         =
            # partial pressure @ surface:
            #   SUR_RH * p0(SUR_T)
            # => SUR_RH = RH * p0(T) / p0(SUR_T)
            self.raw_data["SUR_RH_HR_AVG"] = self.raw_data.apply(
                lambda day: day["RH_HR_AVG"] * water_vapor_presssure(day["T_HR_AVG"]) / \
                        water_vapor_presssure(day["SUR_TEMP"]), axis=1
            )

        def viz_timeseries(
            self, save=False, incl_legend=True, 
            legend_dx=0.0, legend_dy=0.0, plot_lines=False
        ):
            place_to_color = {'air': "k", 'surface': "k"}

            fig, axs = plt.subplots(2, 1, sharex=True)#, figsize=(6.4*0.8, 4.8*.8))
            plt.xticks(rotation=90, ha='center')
            n_days = len(self.wdata["night"]["datetime"])
            # axs[1].xaxis.set_major_locator(
            #     mdates.AutoDateLocator(minticks=n_days-1, maxticks=n_days+1)
            # )

            axs[0].set_title(self.loc_title + f" ({self.year})")

            # T
            if plot_lines:
                axs[0].plot(
                    self.raw_data["datetime"], self.raw_data["T_HR_AVG"], 
                    label="bulk air", color=place_to_color["air"], linewidth=2
                )
                axs[0].plot(
                    self.raw_data["datetime"], self.raw_data["SUR_TEMP"], 
                    label="soil surface", color=place_to_color["surface"], linewidth=2, linestyle="--"
                )
            axs[0].set_ylabel("temperature\n[°C]")
            axs[0].scatter(
                self.wdata["night"]["datetime"], self.wdata["night"]["T_HR_AVG"],
                edgecolors="black", clip_on=False,
                marker="^", color=time_to_color["night"], zorder=10, label="adsorption\nconditions", 
                s=25
            ) # nighttime air temperature
            axs[0].scatter(
                self.wdata["day"]["datetime"], self.wdata["day"]["SUR_TEMP"],
                edgecolors="black", clip_on=False,
                marker="v", color=time_to_color["day"], zorder=10, label="desorption\nconditions",
                s=25
            ) # daytime surface temperature
            # axs[0].set_title(self.location)
            axs[0].set_ylim(self.T_range[0], self.T_range[1])
            axs[0].set_yticks(self.T_ticks)
            axs[0].set_xlim(
                self.raw_data["datetime"].min(), 
                self.raw_data["datetime"].max()
            )

            # RH
            if plot_lines:
                axs[1].plot(
                    self.raw_data["datetime"], self.raw_data["RH_HR_AVG"] / 100, 
                    color=place_to_color["air"], label="bulk air"
                )
                axs[1].plot(
                    self.raw_data["datetime"], self.raw_data["SUR_RH_HR_AVG"] / 100, 
                    color=place_to_color["surface"], label="near-surface air", linestyle="--"
                )
            axs[1].set_ylabel("relative\nhumidity")
            axs[1].scatter(
                self.wdata["night"]["datetime"], self.wdata["night"]["RH_HR_AVG"] / 100,
                edgecolors="black", clip_on=False,
                marker="^", color=time_to_color["night"], zorder=10, 
                s=25,  label="capture conditions"
            ) # nighttime RH
            axs[1].scatter(
                self.wdata["day"]["datetime"], self.wdata["day"]["SUR_RH_HR_AVG"] / 100,
                edgecolors="black", clip_on=False,
                marker="v", color=time_to_color["day"], zorder=10, s=25, label="release conditions"
            ) # day surface RH
            axs[1].set_yticks(self.p_ovr_p0_ticks)
            if self.daynight_wdata.shape[0] > 1:
                axs[1].xaxis.set_major_formatter(my_date_format)
            if incl_legend:
                axs[1].legend(
                    prop={'size': 10}, ncol=1, 
                    bbox_to_anchor=(0., 1.0 + legend_dy, 1.0 + legend_dx, .1), loc="center"
                )#, loc="center left")

            # already got legend above
            if save:
                plt.savefig(self.save_tag + "weather_timeseries.pdf", format="pdf", bbox_inches="tight")

            plt.show()

        def _minimalize_raw_data(self):
            self.raw_data = self.raw_data[["datetime"] + self.relevant_weather_cols]

        def _day_night_data(self):
            # get separate day and night data frames with precise time stamp
            # useful for checking and for plotting as a time series with all of the data
            self.wdata = dict()
            for time in ["day", "night"]:
                self.wdata[time] = self.raw_data[
                    self.raw_data["datetime"].dt.hour == self.time_to_hour[time]
                ]

            ###
            #   create abstract data frame that removes details of the time
            #   each row is a day-night cycle
            ###
            reduced_wdata = dict()
            for time in ["day", "night"]:
                reduced_wdata[time] = self.wdata[time].rename(
                    columns={col: time + "_" + col for col in self.relevant_weather_cols}
                )
                reduced_wdata[time]["datetime"] = reduced_wdata[time]["datetime"].dt.normalize()

            self.daynight_wdata = pd.merge(
                reduced_wdata["night"], reduced_wdata["day"],
                on="datetime", how="inner"
            )

            self.daynight_wdata.sort_values(by="datetime", inplace=True)

        def _gen_ads_des_conditions(self):
            self.ads_des_conditions = self.daynight_wdata.rename(
                columns=
                {
                    "datetime": "date",
                    # adsorptin conditions (night)
                    "night_T_HR_AVG": 'ads T [°C]',
                    "night_RH_HR_AVG": 'ads P/P0',
                    # desorption conditions (day)
                    "day_SUR_TEMP": 'des T [°C]',
                    "day_SUR_RH_HR_AVG": 'des P/P0'
                }
            )
            for rh_col in ['des P/P0', 'ads P/P0']:
                self.ads_des_conditions[rh_col] = self.ads_des_conditions[rh_col] / 100.0

            self.ads_des_conditions = self.ads_des_conditions[
                ['date', 'ads T [°C]', 'ads P/P0', 'des T [°C]', 'des P/P0']
            ]

        def _compute_p_ovr_p0_max(self):
            self.p_ovr_p0_max = np.ceil(
                self.ads_des_conditions[
                    ["ads P/P0", "des P/P0"]
                ].max().max() * 10.0
            ) / 10.0 + 0.1
            self.p_ovr_p0_ticks = np.linspace(
                0, self.p_ovr_p0_max, int(np.ceil(self.p_ovr_p0_max * 10)) + 1
            )
            print("p/p0 max: ", self.p_ovr_p0_max)

        def _compute_T_range(self):
            T_min = self.ads_des_conditions[
                ["ads T [°C]", "des T [°C]"]
            ].min().min()
            T_min = np.floor(T_min / 10) * 10

            T_max = self.ads_des_conditions[
                ["ads T [°C]", "des T [°C]"]
            ].max().max()
            T_max = np.ceil(T_max / 10) * 10

            self.T_range = [T_min, T_max]
            self.T_ticks = np.linspace(
                T_min, T_max, int(np.ceil((T_max - T_min) / 10)) + 1
            )

        def _filter_missing(self):
            print("filtering # missing in raw: ", 
                  np.sum(self.raw_data["T_HR_AVG"] < -999.0)
            )
            self.raw_data = self.raw_data[self.raw_data["T_HR_AVG"] > -999.0]
    return (Weather,)


@app.cell
def _(Weather):
    # weather = Weather([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 2025, "Mercury")
    mos_of_year = list(range(1, 13))
    weather = Weather(mos_of_year, 2025, "Stovepipe")
    weather = Weather([7], 2025, "Stovepipe")
    # weather = Weather(mos_of_year, 2025, "Mercury")


    # weather = Weather([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 2025, "Utqiagvik")

    # weather = Weather([6, 7, 8], 2025, "Utqiagvik")
    weather.ads_des_conditions
    # weather.raw_data
    return (weather,)


@app.cell
def _(my_colors, plt, sns, weather):
    with sns.plotting_context("notebook", font_scale=1.4):
        pp = sns.pairplot(
            weather.ads_des_conditions[1:].rename(
                columns={
                    'ads T [°C]': 'capture $T$ [°C]',
                    'des T [°C]': 'release $T$ [°C]',
                    'ads P/P0': 'capture $p/p_0(T)$',
                    'des P/P0': 'release $p/p_0(T)$',
                }
            ),
            corner=True,
            plot_kws=dict(marker="+", linewidth=1, color=my_colors[0]),
            diag_kws=dict(fill=False, color=my_colors[0]),
            diag_kind='kde'
        )

        pp.axes[1, 1].set_ylim(0, weather.p_ovr_p0_max)
        pp.axes[1, 1].set_yticks(weather.p_ovr_p0_ticks)
        pp.axes[3, 1].set_ylim(0, weather.p_ovr_p0_max)
        pp.axes[3, 1].set_yticks(weather.p_ovr_p0_ticks)
        pp.axes[3, 1].set_xlim(0, weather.p_ovr_p0_max)
        pp.axes[3, 1].set_xticks(weather.p_ovr_p0_ticks)
        pp.axes[3, 3].set_xlim(0, weather.p_ovr_p0_max)
        pp.axes[3, 3].set_xticks(weather.p_ovr_p0_ticks)

        pp.axes[0, 0].set_xlim(weather.T_range)
        pp.axes[0, 0].set_xticks(weather.T_ticks)
        pp.axes[3, 2].set_xlim(weather.T_range)
        pp.axes[3, 2].set_xticks(weather.T_ticks)
        pp.axes[2, 0].set_ylim(weather.T_range)
        pp.axes[2, 0].set_yticks(weather.T_ticks)
    
        plt.tight_layout()
        plt.savefig(
            weather.save_tag + "ads_des_conditions.pdf", 
            format="pdf"
        )
    pp
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # modeling a water adsorption isotherm in a MOF bed
    """)
    return


@app.cell
def _(weather):
    p_over_p0_max = weather.p_ovr_p0_max
    p_over_p0_ticks = weather.p_ovr_p0_ticks
    p_over_p0_max
    return p_over_p0_max, p_over_p0_ticks


@app.cell
def _(math):
    def bern_poly(x, v, n):
        return math.comb(n, v) * x ** v * (1.0 - x) ** (n - v)
    return (bern_poly,)


@app.cell
def _(bern_poly, colors, mpl, np, p_over_p0_max, plt):
    class WaterAdsorptionIsotherm:
        def __init__(
            self, n, Tref=25.0, w_max=0.5, bs=None, p_ovr_p0_max=p_over_p0_max
        ):
            # number of control points
            self.n = n

            # max water ads [kg H2O/kg MOF]
            self.w_max = w_max

            # max RH (at Tref) to model
            self.p_ovr_p0_max = p_ovr_p0_max

            # reference temperature [deg. C]
            self.Tref = Tref

            # pre-allocate bs
            if bs is None:
                self.bs = np.full(n + 1, np.nan)
            else:
                self.bs = bs

        def copy(self):
            return WaterAdsorptionIsotherm(
                self.n, Tref=self.Tref, p_ovr_p0_max=self.p_ovr_p0_max,
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
            # model: expand adsorption n as a function of phi_ref = p / p0[T_ref]
            #        with Bernstein polynomial basis functions.
            # Polanyi: A = - R T log(p / p0[T])
            #          n = n(A)
            # set A = - RT log(phi) = - R T_ref log(phi_ref)
            #     cuz we wanna know corresponding phi_ref at T_ref that gives same A at T
            #        T / T_Ref log(phi) = log(phi_ref)
            #        log(phi^(T/T_Ref)) = log(phi_ref) 
            p_over_p0_ref = p_over_p0 ** ((T + 273.15) /  (self.Tref + 273.15))

            if p_over_p0_ref > self.p_ovr_p0_max:
                return self.w_max

            a = 0.0 # amount adsorbed [unitless]
            x = p_over_p0_ref / self.p_ovr_p0_max
            for v in range(self.n + 1):
                a += self.bs[v] * bern_poly(x, v, self.n)

            return a

        def water_del(self, conditions):
            w_del = np.zeros(conditions.shape[0])
            for i, (id, row) in enumerate(conditions.iterrows()):
                w_ads = self.water_ads(row["ads T [°C]"], row["ads P/P0"])
                w_des = self.water_ads(row["des T [°C]"], row["des P/P0"])
                if w_ads > w_des:
                    w_del[i] = w_ads - w_des
            return w_del

        def water_del_distn(self, weather):
            w_dels = self.water_del(weather.ads_des_conditions)

            plt.figure()
            plt.hist(w_dels)
            plt.ylabel("# days")
            plt.xlabel("water delivery")
            plt.show()

        def draw(self):
            p_over_p0s = np.linspace(0, self.p_ovr_p0_max, 100)

            plt.figure()

            plt.xlabel("relative humidity $p / [p_0(T)]$")
            plt.ylabel("water adsorption [kg H$_2$O/kg MOF]")

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
            plt.xlim(0, self.p_ovr_p0_max)
            plt.ylim(0, self.w_max)

            plt.show()
    return (WaterAdsorptionIsotherm,)


@app.cell
def _(WaterAdsorptionIsotherm, plt):
    wai = WaterAdsorptionIsotherm(10, p_ovr_p0_max=0.5)
    wai.endow_stepped_isotherm(3)
    wai.draw()
    plt.tight_layout()
    plt.show()
    return (wai,)


@app.cell
def _(weather):
    weather.viz_timeseries(save=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # dist'n of water deliveries
    """)
    return


@app.cell
def _(np):
    # value at risk https://en.wikipedia.org/wiki/Value_at_risk
    def score_fitness(wai, weather, alpha=10.0):
        # get dist'n of water dels
        water_dels = wai.water_del(weather.ads_des_conditions)
        # get worst-case water delivery, ignoring alpha % of hard cases.
        return np.percentile(water_dels, alpha)
    return (score_fitness,)


@app.cell
def _(score_fitness, wai, weather):
    fitness = score_fitness(wai, weather)
    fitness
    return (fitness,)


@app.cell
def _(fitness, plt, wai, weather):
    plt.figure()
    plt.hist(wai.water_del(weather.ads_des_conditions))
    plt.axvline(fitness, color="C1", label=f"10% VAR")
    plt.ylabel("# days")
    plt.xlim(0, 1)
    plt.legend()
    plt.xlabel("water delivery")
    plt.tight_layout()
    plt.savefig(weather.save_tag + "eg_var.pdf", format="pdf")
    plt.show()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # random WAIs to explore
    """)
    return


@app.cell
def _(draw_rh_distn, my_colors, np, p_over_p0_ticks, plt, score_fitness):
    def compare_wais(wais, weather, savetag=""):
        the_colors = [my_colors[0]] + my_colors[3:]
        p_over_p0s = np.linspace(0, weather.p_ovr_p0_max, 100)

        fig = plt.figure(figsize=(6, 4.5), layout="constrained")
        gs = fig.add_gridspec(2, 2, height_ratios=[1, 3], width_ratios=[1, 1])
        ax00 = fig.add_subplot(gs[0, 0])
        ax10 = fig.add_subplot(gs[1, 0], sharex=ax00) # Only these two share
        ax01 = fig.add_subplot(gs[0, 1])
        ax11 = fig.add_subplot(gs[1, 1])
        axs = np.array([[ax00, ax01],
                        [ax10, ax11]])

        axs[0, 1].axis('off')

        ###
        #   adsorption isotherm
        ###
        axs[1, 0].set_xlabel("$p / [p_0(T)]$")
        axs[1, 0].set_xticks(weather.p_ovr_p0_ticks)
        axs[1, 0].set_ylabel(
            f"water adsorption at {wais[0].Tref:.0f}°C\n[kg H$_2$O/kg MOF]"
        )

        for w, wai in enumerate(wais):
            axs[1, 0].plot(
                p_over_p0s, 
                [wai.water_ads(wai.Tref, p_over_p0) for p_over_p0 in p_over_p0s],
                color=the_colors[w],
                label=f"#{w}"
            )

        axs[1, 0].set_xlim(0, weather.p_ovr_p0_max)
        axs[1, 0].set_ylim(0, wais[0].w_max)
        axs[1, 0].legend(title="model material", fontsize=8, title_fontsize=10)

        ###
        #   P/P0 distns
        ###
        draw_rh_distn(axs[0, 0], weather)

        ###
        #   working cap dist'n
        ###
        bins = np.linspace(0, 0.5, 12)
        for w, wai in enumerate(wais):
            fitness = score_fitness(wai, weather)

            axs[1, 1].hist(
                wai.water_del(weather.ads_des_conditions),
                edgecolor=the_colors[w], histtype='step',
                bins=bins
            )
            axs[1, 1].hist(
                wai.water_del(weather.ads_des_conditions),
                color=the_colors[w], alpha=0.25,
                bins=bins
            )

            axs[1, 1].axvline(
                fitness, color=the_colors[w], linestyle="--"
            )
        axs[1, 1].set_ylabel("# days")
        axs[1, 1].set_yticks([0, 100, 200])
        axs[1, 1].set_xticks(p_over_p0_ticks)
        axs[1, 1].set_ylim(0, 300)
        axs[1, 1].set_xlabel("water delivery\n[kg H$_2$O/kg MOF]")
        # axs[1, 1].legend(fontsize=12)

        # fitness label:
        fitness_label = f"fitness:\n{fitness:.2f} kg H$_2$O/kg MOF",

        plt.savefig(
            weather.save_tag + "compare" + savetag + ".pdf",
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
    # evolutionary optimization
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 🦋 evolutionary operations
    """)
    return


@app.cell
def _(my_colors, np, p_over_p0_ticks, plt):
    def viz_wais(
        wais, savename=None, material_labels=None
    ):
        if material_labels is None:
            material_labels = [f"#{w}" for w in range(len(wais))]

        the_colors = [my_colors[0]] + my_colors[3:]
        p_over_p0s = np.linspace(0, wais[0].p_ovr_p0_max, 100)

        fig = plt.figure()
        plt.xlabel("$p / [p_0(T)]$")
        plt.xticks(p_over_p0_ticks)
        plt.ylabel(
            f"water adsorption at {wais[0].Tref:.0f}°C\n[kg H$_2$O/kg MOF]"
        )

        for w, wai in enumerate(wais):
            plt.plot(
                p_over_p0s, 
                [wai.water_ads(wai.Tref, p_over_p0) for p_over_p0 in p_over_p0s],
                color=the_colors[w],
                label=material_labels[w]
            )

        plt.xlim(0, wais[0].p_ovr_p0_max)
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
        [random_birth(n) for i in range(4)], savename="random_births"
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
        delta_b = 2 * eps * np.sort(np.random.rand(wai.n - 1) - 0.5)
        wai.bs[1:-1] += delta_b

        # enforce constraint
        wai.bs[wai.bs < 0.0] = 0.0
        wai.bs[wai.bs > wai.w_max] = wai.w_max
        wai.bs[-1] = wai.w_max
    return (mutate,)


@app.cell
def _(WaterAdsorptionIsotherm, mutate, viz_wais):
    _wais = [WaterAdsorptionIsotherm(10)]
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
    _rand_wais = [WaterAdsorptionIsotherm(10), WaterAdsorptionIsotherm(10)]
    _rand_wais[0].endow_stepped_isotherm(4)
    _rand_wais[1].endow_stepped_isotherm(8)
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
    _rand_wais = [WaterAdsorptionIsotherm(10), WaterAdsorptionIsotherm(10)]
    _rand_wais[0].endow_stepped_isotherm(2)
    _rand_wais[1].endow_random_isotherm()

    _rand_wais.append(random_cross_over(_rand_wais[0], _rand_wais[1]))
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
    if run_evol_cbox.value:
        pop_size = 50
        n_generations = 25
        n = 30
        fitnesses_gen, best_wai_gen, best_wai, best_fitness = do_evolution(
            weather, n_generations, pop_size, n
        )
    return best_fitness, best_wai, best_wai_gen, fitnesses_gen, n


@app.cell
def _(best_wai):
    best_wai.draw()
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
            columns=['generation', 'fitness [kg H$_2$O/kg MOF]']
        )

        sns.stripplot(
            data, 
            x="generation", y="fitness [kg H$_2$O/kg MOF]",
            hue="generation", color="C2", palette="crest", legend=False
        )
        # plt.axhline(
        #     y=step_fitnesses[id_opt_step], 
        #     color="gray", linestyle="--", zorder=-1
        # )
        plt.tight_layout()
        plt.savefig(
            weather.save_tag + "fitness_progress.pdf", format="pdf"
        )
        plt.show()

    viz_fitness_progress(fitnesses_gen)
    return


@app.cell
def _(best_wai_gen, colors, mpl, np, plt, weather):
    def viz_best_wais(best_wai_gen):
        p_over_p0s = np.linspace(0, best_wai_gen[0].p_ovr_p0_max, 150)
        Tref = best_wai_gen[0].Tref

        colormap = mpl.colormaps['crest'] # or 'plasma', 'coolwarm', etc.
        norm = colors.Normalize(vmin=0, vmax=len(best_wai_gen))

        plt.figure()
        plt.xlabel("$p/p_0[T]$")
        plt.ylabel("water adsorption [kg H$_2$O/kg MOF]")
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
            [0.7, 0.12, 0.2, 0.6]
        )
        cb_ax.axis("off")
        plt.colorbar(
            sm, ax=cb_ax, label='generation', 
        )

        plt.tight_layout()
        plt.savefig(
            weather.save_tag + "wai_progress.pdf", format="pdf"
        )
        plt.show()

    viz_best_wais(best_wai_gen)
    return


@app.cell
def _(best_wai, plt, weather):
    best_wai.draw()
    plt.tight_layout()
    plt.savefig(
        weather.save_tag + "best_wai.pdf", format="pdf"
    )
    plt.show()
    return


@app.cell
def _(best_wai, plt, sns, weather):
    def viz_daily_performance(wai, weather):
        # TODO plot delta p/P0, delta T
        conditions = weather.ads_des_conditions.copy()
        conditions["water delivery [kg H$_2$O/kg MOF]"] = wai.water_del(
            weather.ads_des_conditions
        )

        # Initialize the grid
        pp = sns.PairGrid(
            conditions, hue="water delivery [kg H$_2$O/kg MOF]", corner=True
        )

        # Map only to the off-diagonal (lower) plots
        pp.map_lower(sns.scatterplot)

        # Optional: Add a legend since we are using PairGrid manually
        handles, labels = pp.axes[1, 0].get_legend_handles_labels()

        pp.fig.legend(
            handles, 
            labels,
            title="water delivery [kg H$_2$O/kg MOF]",
            loc="upper right", 
            bbox_to_anchor=(0.8, 0.8) 
        )

        pp.axes[1, 1].set_ylim(0, weather.p_ovr_p0_max)
        pp.axes[1, 1].set_yticks(weather.p_ovr_p0_ticks)
        pp.axes[3, 1].set_ylim(0, weather.p_ovr_p0_max)
        pp.axes[3, 1].set_yticks(weather.p_ovr_p0_ticks)
        pp.axes[3, 1].set_xlim(0, weather.p_ovr_p0_max)
        pp.axes[3, 1].set_xticks(weather.p_ovr_p0_ticks)
        pp.axes[3, 3].set_xlim(0, weather.p_ovr_p0_max)
        pp.axes[3, 3].set_xticks(weather.p_ovr_p0_ticks)

        pp.axes[0, 0].set_xlim(weather.T_range)
        pp.axes[0, 0].set_xticks(weather.T_ticks)
        pp.axes[3, 2].set_xlim(weather.T_range)
        pp.axes[3, 2].set_xticks(weather.T_ticks)
        pp.axes[2, 0].set_ylim(weather.T_range)
        pp.axes[2, 0].set_yticks(weather.T_ticks)

        for i in range(4):
            pp.axes[i, i].set_visible(False)


        plt.savefig(
            weather.save_tag + "daily_performance.pdf", format="pdf",
            bbox_inches="tight"
        )

        plt.show()

    viz_daily_performance(best_wai, weather)
    return


@app.cell
def _(np, time_to_color):
    def draw_rh_distn(ax, weather):
        p_over_p0_bins = np.linspace(0, 1, 25)
        ax.hist(
            weather.ads_des_conditions["des P/P0"], label="capture", 
            bins=p_over_p0_bins, histtype='step', 
            edgecolor=time_to_color["night"]
        )
        ax.hist(
            weather.ads_des_conditions["des P/P0"],  
            bins=p_over_p0_bins, 
            color=time_to_color["night"], alpha=0.25
        )

        ax.hist(
            weather.ads_des_conditions["ads P/P0"], 
            label="release", histtype='step',
            bins=p_over_p0_bins, edgecolor=time_to_color["day"]
        )
        ax.hist(
            weather.ads_des_conditions["ads P/P0"], 
            bins=p_over_p0_bins, 
            color=time_to_color["day"], alpha=0.25
        )

        ax.set_ylabel("# days")
        ax.set_yticks([0, 100, 200])
        ax.set_ylim(0, 200)
        ax.legend(fontsize=12)
    return (draw_rh_distn,)


@app.cell
def _(
    colors,
    draw_rh_distn,
    mpl,
    my_colors,
    np,
    p_over_p0_ticks,
    plt,
    score_fitness,
):
    def draw_opt(best_wai, weather, savetag=""):
        p_over_p0s = np.linspace(0, best_wai.p_ovr_p0_max, 100)

        # fig, axs = plt.subplots(
        #     2, 2, 
        #     gridspec_kw={'height_ratios': [1, 3], 'width_ratios': [2, 1]},
        #     figsize=(5, 7),
        #     layout="constrained"
        # )
        fig = plt.figure(figsize=(5, 5), layout="constrained")
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
        axs[1, 0].set_xlabel("$p / [p_0(T)]$")
        axs[1, 0].set_xticks(p_over_p0_ticks)
        axs[1, 0].set_ylabel("water adsorption [kg H$_2$O/kg MOF]")

        colormap = mpl.colormaps['coolwarm'] # or 'plasma', 'coolwarm', etc.
        norm = colors.Normalize(vmin=10.0, vmax=60.0)

        for T in np.linspace(10, 60, 4):
            axs[1, 0].plot(
                p_over_p0s, 
                [best_wai.water_ads(T, p_over_p0) for p_over_p0 in p_over_p0s],
                color=colormap(norm(T))
            )

        sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
        # sm.set_array([]) # Required for matplotlib versions < 3.4
        cb_ax = axs[1, 0].inset_axes(
            [0.5, 0.1, 0.2, 0.6]
        )
        cb_ax.axis("off")
        plt.colorbar(
            sm, ax=cb_ax, label='temperature [°C]', 
            # orientation="horizontal"
        )
        axs[1, 0].set_xlim(0, best_wai.p_ovr_p0_max)
        axs[1, 0].set_ylim(0, best_wai.w_max)

        ###
        #   P/P0 distns
        ###
        draw_rh_distn(axs[0, 0], weather)

        ###
        #   working cap dist'n
        ###
        fitness = score_fitness(best_wai, weather)
        print("fitness: ", fitness)

        axs[1, 1].hist(
            best_wai.water_del(weather.ads_des_conditions),
            orientation='horizontal', 
            edgecolor=my_colors[4], histtype='step'
        )
        axs[1, 1].hist(
            best_wai.water_del(weather.ads_des_conditions),
            orientation='horizontal', 
            color=my_colors[4], alpha=0.25
        )
        axs[1, 1].axhline(
            fitness, color="black", linestyle="--",
            label=f"fitness:\n{fitness:.2f}"
        )
        axs[1, 1].set_xlabel("# days")
        axs[1, 1].set_xticks([0, 100, 200])
        axs[1, 1].set_xlim(0, 200)
        axs[1, 1].set_ylabel("water delivery [kg H$_2$O/kg MOF]")
        # axs[1, 1].legend(fontsize=12)

        # fitness label:
        fitness_label = f"fitness:\n{fitness:.2f} kg H$_2$O/kg MOF",

        plt.savefig(
            weather.save_tag + "best_wai_rich" + savetag + ".pdf",
            format="pdf",  bbox_inches="tight"
        )

        plt.show()
    return (draw_opt,)


@app.cell
def _(best_wai, draw_opt, weather):
    draw_opt(best_wai, weather)
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
        return wais, fitnesses, id_opt, opt_fitness

    step_wais, step_fitnesses, id_opt_step, best_fitness_step = search_step_wais(n)
    return best_fitness_step, id_opt_step, step_fitnesses, step_wais


@app.cell
def _(colors, id_opt_step, mpl, np, plt, step_fitnesses, step_wais, weather):
    def viz_step_wais(step_wais, step_fitnesses, id_opt_step):
        Tref = step_wais[0].Tref
        w_max = step_wais[0].w_max

        p_over_p0s = np.linspace(0, step_wais[0].p_ovr_p0_max, 100)

        plt.figure()

        plt.xlabel("relative humidity $p / [p_0(T)]$")
        plt.ylabel("water adsorption [kg H$_2$O/kg MOF]")

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
        plt.xlim(0, step_wais[0].p_ovr_p0_max)
        plt.ylim(0, w_max)

        plt.savefig(
            weather.save_tag + "step_search.pdf",
            format="pdf",  bbox_inches="tight"
        )

        plt.show()

    viz_step_wais(step_wais, step_fitnesses, id_opt_step)
    return


@app.cell
def _(best_fitness, best_fitness_step):
    print(
        "mass savings over a step: ",
        (best_fitness - best_fitness_step) / best_fitness_step
    )
    return


@app.cell
def _(draw_opt, id_opt_step, step_wais, weather):
    draw_opt(step_wais[id_opt_step], weather, savetag="baseline")
    return


if __name__ == "__main__":
    app.run()
