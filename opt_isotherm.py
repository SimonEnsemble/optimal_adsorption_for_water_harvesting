import marimo

__generated_with = "0.20.2"
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
    return colors, math, mo, mpl, my_date_format, np, os, pd, plt, sns


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
    # modeling a water adsorption isotherm in a MOF bed
    """)
    return


@app.cell
def _(math):
    def bern_poly(x, v, n):
        return math.comb(n, v) * x ** v * (1.0 - x) ** (n - v)

    return (bern_poly,)


@app.cell
def _(bern_poly, colors, mpl, np, plt):
    class WaterAdsorptionIsotherm:
        def __init__(self, n, Tref=25.0, w_max=0.5):
            # number of control points
            self.n = n

            # max water ads [kg H2O/kg MOF]
            self.w_max = w_max

            # reference temperature [deg. C]
            self.Tref = Tref

            # pre-allocate bs
            self.bs = np.full(n + 1, np.nan)

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

            a = 0.0 # amount adsorbed [unitless]
            for v in range(self.n + 1):
                a += self.bs[v] * bern_poly(p_over_p0_ref, v, self.n)
            return a

        def water_del(self, conditions):
            w_ads = self.water_ads(conditions["ads T [°C]"], conditions["ads P/P0"])
            w_des = self.water_ads(conditions["des T [°C]"], conditions["des P/P0"])
            w_del = w_ads - w_des
            return np.maximum(0, w_del.values) # can't be negative

        def water_del_distn(self, weather):
            w_dels = self.water_del(weather.ads_des_conditions)

            plt.figure()
            plt.hist(w_dels)
            plt.ylabel("# days")
            plt.xlabel("water delivery")
            plt.show()

        def draw(self):
            p_over_p0s = np.linspace(0, 1, 100)

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
            plt.xlim(0, 1)
            plt.ylim(0, self.w_max)

            plt.show()

    return (WaterAdsorptionIsotherm,)


@app.cell
def _(WaterAdsorptionIsotherm, fig_dir, plt):
    wai = WaterAdsorptionIsotherm(10)
    wai.endow_stepped_isotherm(2)
    wai.draw()
    plt.tight_layout()
    plt.savefig(fig_dir + f"/eg_wai.pdf", format="pdf")
    plt.show()
    return (wai,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # data on capture and release conditions

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
            axs[0].set_ylim(-5, 75)
            axs[0].set_yticks([-10 + 10 * _i for _i in range(8)])
            axs[0].set_xlim(self.raw_data["datetime"].min(), self.raw_data["datetime"].max())

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
            axs[1].set_yticks([0.2 * _i for _i in range(6)])
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
    # weather = Weather(mos_of_year, 2025, "Mercury")


    # weather = Weather([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 2025, "Utqiagvik")

    # weather = Weather([6, 7, 8], 2025, "Utqiagvik")
    weather.ads_des_conditions
    # weather.raw_data
    return (weather,)


@app.cell
def _(my_colors):
    my_colors
    return


@app.cell
def _(my_colors, np, plt, sns, weather):
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
        pp.axes[1, 1].set_ylim(0, 1)
        pp.axes[1, 1].set_yticks(np.linspace(0, 1, 6))
        pp.axes[3, 1].set_ylim(0, 1)
        pp.axes[3, 1].set_yticks(np.linspace(0, 1, 6))
        pp.axes[3, 1].set_xlim(0, 1)
        pp.axes[3, 1].set_xticks(np.linspace(0, 1, 6))
        pp.axes[3, 3].set_xlim(0, 1)
        pp.axes[3, 3].set_xticks(np.linspace(0, 1, 6))

        T_range = [-10, 75]
        assert T_range[1] > weather.ads_des_conditions[["ads T [°C]", "des T [°C]"]].max().max()
        assert T_range[0] < weather.ads_des_conditions[["ads T [°C]", "des T [°C]"]].min().min()

        T_ticks = [-10 + 20*i for i in range(5)]

        pp.axes[0, 0].set_xlim(T_range[0], T_range[1])
        pp.axes[0, 0].set_xticks(T_ticks)
        pp.axes[3, 2].set_xlim(T_range[0], T_range[1])
        pp.axes[3, 2].set_xticks(T_ticks)
        pp.axes[2, 0].set_ylim(T_range[0], T_range[1])
        pp.axes[2, 0].set_yticks(T_ticks)
        plt.tight_layout()
        plt.savefig(
            weather.save_tag + "ads_des_conditions.pdf", 
            format="pdf"
        )
    pp
    return


@app.cell
def _(weather):
    weather.viz_timeseries(save=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## dist'n of water deliveries
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
    ## evolutionary optimization
    """)
    return


@app.cell
def _(WaterAdsorptionIsotherm, np, score_fitness, weather):
    def evolve(
        wais, n_elite=5, tourney_size=10, 
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

        if verbose:
            print("initial generation")
            print("\telite fitness: ", fitnesses[ids_elite])

        # initiate new generation with the elite individuals un-modified
        new_wais = [wais[i_elite] for i_elite in ids_elite]

        # tournament selection
        for i in range(pop_size - n_elite - n_rand):
            # pick individuals for tournament
            ids_tourney = np.random.choice(
                range(pop_size), size=tourney_size, replace=False
            )

            # compete for top two (= the chosen parents)
            ids_winners = np.argpartition(fitnesses[ids_tourney], -2)[-2:]
            id_a = ids_tourney[ids_winners[0]]
            id_b = ids_tourney[ids_winners[1]]

            # mate to produce child
            wai = WaterAdsorptionIsotherm(dim)
            alpha = np.random.rand() # fraction of genes of parent a
            wai.bs = alpha * wais[id_a].bs + (1 - alpha) * wais[id_b].bs
            new_wais.append(wai)

        # random births for exploration
        for i in range(n_rand):
            wai = WaterAdsorptionIsotherm(dim)
            if np.random.rand() < 0.5:
                wai.endow_random_stepped_isotherm()
            else:
                wai.endow_random_isotherm()
            new_wais.append(wai)

        # mutation
        for i in range(n_mutate):
            # select non-elite individual to mutate
            id = np.random.choice(np.arange(n_elite, pop_size))

            # mutation
            delta_b = 2 * eps * np.sort(np.random.rand(dim - 1) - 0.5)

            new_wais[id].bs[1:-1] += delta_b
            new_wais[id].bs[new_wais[id].bs < 0.0] = 0.0
            new_wais[id].bs[new_wais[id].bs > w_max] = w_max
            new_wais[id].bs[-1] = w_max

        return new_wais

    return (evolve,)


@app.cell
def _(WaterAdsorptionIsotherm, np):
    def gen_initial_pop(pop_size, dim):
        wais = [WaterAdsorptionIsotherm(dim) for _ in range(pop_size)]
        for wai in wais:
            if np.random.rand() < 0.5:
                wai.endow_random_stepped_isotherm()
            else:
                wai.endow_random_isotherm()
        return wais

    return (gen_initial_pop,)


@app.cell
def _(evolve, gen_initial_pop, np, plt, score_fitness, weather):
    # first generation
    wais = gen_initial_pop(75, 25)

    fitnesses = np.array([score_fitness(wai, weather) for wai in wais])

    # second generation
    new_wais = evolve(wais, n_elite=5)
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
def _(evolve, gen_initial_pop, np, score_fitness, weather):
    def do_evolution(n_generations, pop_size, dim):
        # generate population
        wais = gen_initial_pop(pop_size, dim)

        # score fitnesses
        fitnesses = np.array([score_fitness(wai, weather) for wai in wais])

        # store progress
        fitnesses_gen = [fitnesses]
        best_wai_gen = [wais[np.argmax(fitnesses)]]

        # evolve over generations
        for g in range(1, n_generations):
            wais = evolve(wais)
            fitnesses = np.array([score_fitness(wai, weather) for wai in wais])

            fitnesses_gen.append(fitnesses)
            best_wai_gen.append(wais[np.argmax(fitnesses)])

        best_wai = wais[np.argmax(fitnesses)]

        return fitnesses_gen, best_wai_gen, best_wai

    return (do_evolution,)


@app.cell
def _(do_evolution, run_evol_cbox):
    if run_evol_cbox.value:
        pop_size = 65
        n_generations = 25
        dim = 30
        fitnesses_gen, best_wai_gen, best_wai = do_evolution(
            n_generations, pop_size, dim
        )
    return best_wai, best_wai_gen, dim, fitnesses_gen


@app.cell
def _(WaterAdsorptionIsotherm, np, score_fitness):
    # increase capacity at high pressure until fitness decreases
    def top_off(best_wai, weather):
        new_best_wai = WaterAdsorptionIsotherm(best_wai.n)
        new_best_wai.bs = np.copy(best_wai.bs)

        fitness = score_fitness(best_wai, weather)
        print("current fitness: ", fitness)
        for i in range(best_wai.n):
            new_wai = WaterAdsorptionIsotherm(best_wai.n)
            new_wai.bs = np.copy(new_best_wai.bs)
            new_wai.bs[i:] = best_wai.w_max
            new_fitness = score_fitness(new_wai, weather)
            if new_fitness >= fitness: # OR EQUAL TO (important)
                print(
                    "increased uptake at high p/p0 w./o decrease in fitness."
                )
                print("\tnew fitness: ", new_fitness)
                new_best_wai.bs = new_wai.bs
                fitness = new_fitness
        return new_best_wai   

    return (top_off,)


@app.cell
def _(best_wai, top_off, weather):
    # local search to improve
    best_wai_ls = top_off(best_wai, weather)
    best_wai_ls.draw()
    return (best_wai_ls,)


@app.cell
def _(baseline_fitnesses, fitnesses_gen, id_opt_baseline, plt, sns, weather):
    sns.stripplot(fitnesses_gen, color="C2", palette="crest")
    plt.xlabel("generation")
    plt.ylabel("fitness [kg H$_2$O/kg MOF]")
    plt.axhline(
        y=baseline_fitnesses[id_opt_baseline], color="black", linestyle="--"
    )
    plt.tight_layout()
    plt.savefig(
        weather.save_tag + "fitness_progress.pdf", format="pdf"
    )
    plt.show()
    return


@app.cell
def _(best_wai_gen, colors, mpl, np, plt, weather):
    def viz_best_wais(best_wai_gen):
        p_over_p0s = np.linspace(0, 1, 150)
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
def _(colors, mpl, my_colors, np, plt, score_fitness, time_to_color):
    def draw_opt(best_wai, weather, savetag=""):
        p_over_p0s = np.linspace(0, 1, 100)

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
        axs[1, 0].set_xticks(np.linspace(0, 1, 6))
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
        axs[1, 0].set_xlim(0, 1)
        axs[1, 0].set_ylim(0, best_wai.w_max)

        ###
        #   P/P0 distns
        ###
        p_over_p0_bins = np.linspace(0, 1, 25)
        axs[0, 0].hist(
            weather.ads_des_conditions["des P/P0"], label="capture", 
            bins=p_over_p0_bins, histtype='step', 
            edgecolor=time_to_color["night"]
        )
        axs[0, 0].hist(
            weather.ads_des_conditions["ads P/P0"], 
            label="release", histtype='step',
            bins=p_over_p0_bins, edgecolor=time_to_color["day"]
        )

        axs[0, 0].set_ylabel("# days")
        axs[0, 0].set_yticks([0, 100, 200])
        axs[0, 0].set_ylim(0, 200)
        axs[0, 0].legend(fontsize=12)

        ###
        #   working cap dist'n
        ###
        fitness = score_fitness(best_wai, weather)

        axs[1, 1].hist(
            best_wai.water_del(weather.ads_des_conditions),
            orientation='horizontal', 
            edgecolor=my_colors[4], histtype='step'
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
def _(best_wai_ls, draw_opt, weather):
    draw_opt(best_wai_ls, weather)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## baseline: a stepped adsorption isotherm
    search for best stepped adsorption isotherm
    """)
    return


@app.cell
def _(WaterAdsorptionIsotherm, dim, np, score_fitness, weather):
    n_grid = dim
    wai_all_steps = [WaterAdsorptionIsotherm(n_grid) for i in range(n_grid)]
    for n_step in np.arange(n_grid):
        wai_all_steps[n_step].endow_stepped_isotherm(n_step)

    baseline_fitnesses = np.array(
        [score_fitness(wai, weather) for wai in wai_all_steps]
    )
    id_opt_baseline = np.argmax(baseline_fitnesses)
    return baseline_fitnesses, id_opt_baseline, wai_all_steps


@app.cell
def _(draw_opt, id_opt_baseline, wai_all_steps, weather):
    draw_opt(wai_all_steps[id_opt_baseline], weather, savetag="baseline")
    return


if __name__ == "__main__":
    app.run()
