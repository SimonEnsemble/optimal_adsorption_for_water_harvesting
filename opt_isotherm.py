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

    # date format
    my_date_format_str = '%b-%d'
    my_date_format = mdates.DateFormatter(my_date_format_str)
    return (
        colors,
        math,
        mdates,
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # modeling the vapor pressure of water
    """)
    return


@app.cell
def _(np, warnings):
    # input  T  : deg C
    # output P* : bar
    def water_vapor_presssure(T):
        if T < 273 - 273.15 or T > 343 - 273.15:
            warnings.warn(f"{T}°C outside T range of Antoinne eqn.")
        A = B = C = np.nan
        # coefficients for the following setup:
        #  log10(P) = A − (B / (T + C))
        #     P = vapor pressure (bar)
        #     T = temperature (K)
        if T > 293 - 273.15:
            # valid for 293. to 343 K
            A = 6.20963
            B = 2354.731
            C = 7.559
        if T < 293 - 273.15: # cover a bit lower temperatures
            # valid for 273. to 303 K
            A = 5.40221
            B = 1838.675
            C = -31.737
        return 10.0 ** (A - B / ((T + 273.15) + C))
    return (water_vapor_presssure,)


@app.cell
def _(np, plt, water_vapor_presssure):
    def viz_water_vapor_presssure():
        Ts = np.linspace(273, 343, 100) - 273.15 # deg C

        plt.figure()
        plt.xlabel("T [°C]")
        plt.ylabel("P* [bar]")
        plt.plot(Ts, [water_vapor_presssure(T_i) for T_i in Ts], linewidth=3)
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
def _():
    if None:
        print("yes")
    return


@app.cell
def _(bern_poly, colors, mpl, np, plt):
    class WaterAdsorptionIsotherm:
        def __init__(self, n, Tref=25.0, method="", n_step=None):
            # number of control points
            self.n = n

            # reference temperature [deg. C]
            self.Tref = Tref
            
            # weights on Bernstein polynomials
            viable_methods = ["uniform", "random step", "manual step"]
            if method == "uniform":
                self.bs = np.sort(np.random.rand(n+1))
            elif method == "random step":
                self.bs = np.zeros(n + 1)
                i = np.random.choice(n)
                self.bs[i:] = 1.0
            elif method == "manual step" and n_step:
                self.bs = np.zeros(n + 1)
                self.bs[n_step:] = 1.0
            else:
                raise Exception(
                    f"method not implemented. choose a viable method: {viable_methods}. if manual step, provide n_step"
                )

            # no matter what
            self.bs[0] = 0.0 # start at zero
            self.bs[-1] = 1.0 # end at 1

        def water_ads(self, T, p_over_p0):
            # p/p0 at Tref that gives same Polanyi potential
            p_over_p0_ref = p_over_p0 ** ((T + 273.15) /  (self.Tref + 273.15))

            a = 0.0
            for v in range(self.n+1):
                a += self.bs[v] * bern_poly(p_over_p0_ref, v, self.n)
            return a

        def water_del(self, ads_des_conditions):
            w_ads = self.water_ads(
                ads_des_conditions["ads T [°C]"], ads_des_conditions["ads P/P0"]
            )
            w_des = self.water_ads(
                ads_des_conditions["des T [°C]"], ads_des_conditions["des P/P0"]
            )
            w_del = w_ads - w_des
            w_del = np.maximum(0, w_del.values) # can't be negative
            return w_del

        def water_del_distn(self, weather):
            plt.figure()
            plt.hist(self.water_del(weather.ads_des_conditions))
            plt.ylabel("# days")
            plt.xlabel("water delivery")
            plt.show()

        def draw(self):
            p_over_p0s = np.linspace(0, 1, 100)

            plt.figure()

            plt.xlabel("$p / [p_0(T)]$")
            plt.ylabel("water adsorption")

            colormap = mpl.colormaps['coolwarm'] # or 'plasma', 'coolwarm', etc.
            norm = colors.Normalize(vmin=10.0, vmax=60.0)

            for T in np.linspace(10, 60, 4):
                plt.plot(
                    p_over_p0s, 
                    [self.water_ads(T, p_over_p0) for p_over_p0 in p_over_p0s],
                    color=colormap(norm(T))
                )

            sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
            # sm.set_array([]) # Required for matplotlib versions < 3.4
            plt.colorbar(sm, ax=plt.gca(), label='temperature [°C]')
            plt.xlim(0, 1)
            plt.ylim(0, 1)

            plt.show()
    return (WaterAdsorptionIsotherm,)


@app.cell
def _(weather):
    weather.ads_des_conditions
    return


@app.cell
def _(WaterAdsorptionIsotherm):
    wai = WaterAdsorptionIsotherm(25, method="random step")
    wai.draw()
    return (wai,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # data on capture and release conditions
    """)
    return


@app.cell
def _():
    city_to_state = {'Tucson': 'AZ', 'Socorro': 'NM'}
    return (city_to_state,)


@app.cell
def _():
    time_to_color = {'day': "C1", "night": "C4"}
    time_to_color["ads"] = time_to_color["night"]
    time_to_color["des"] = time_to_color["day"]
    return (time_to_color,)


@app.cell
def _(
    city_to_state,
    fig_dir,
    mdates,
    my_date_format,
    np,
    os,
    pd,
    plt,
    time_to_color,
    water_vapor_presssure,
):
    class Weather:
        def __init__(
            self, months, year, location, time_to_hour={'day': 15, 'night': 5}
        ):
            self.months = months
            self.year = year
            self.location = location

            print(f"reading {year} {location} weather.")
            print("\tnighttime adsorption hr: ", time_to_hour["night"])
            print("\tdaytime harvest hr: ", time_to_hour["day"])

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
            # _start_date = self.ads_des_conditions["date"].min().strftime(my_date_format_str)# .date()
            # _end_data = self.ads_des_conditions["date"].max().strftime(my_date_format_str)# .date()
            self.loc_title = f"{self.location}, {city_to_state[self.location]}."
            # self.loc_timespan_title = f"{self.loc_title} {_start_date} to {_end_data}."

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

            col_names = open(wdata_dir + "/headers.txt", "r").readlines()[1].split()

            raw_data = pd.read_csv(
                wdata_dir + "/" + filename, sep=",", names=col_names, 
                dtype={'LST_DATE': str}
            )

            self.raw_data = raw_data

        def _process_datetime_and_filter(self):
            # convert to pandas datetime
            self.raw_data["date"] = pd.to_datetime(self.raw_data["LST_DATE"])

            # keep only self.month of 2024
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
            self.raw_data["SUR_RH_HR_AVG"] = self.raw_data.apply(
                lambda day: day["RH_HR_AVG"] * water_vapor_presssure(day["T_HR_AVG"]) / \
                        water_vapor_presssure(day["SUR_TEMP"]), axis=1
            )

        def viz_timeseries(self, save=False, incl_legend=True, legend_dx=0.0, legend_dy=0.0, toy=False):
            place_to_color = {'air': "k", 'surface': "k"}

            fig, axs = plt.subplots(2, 1, sharex=True)#, figsize=(6.4*0.8, 4.8*.8))
            if toy:
                for ax in axs:
                    ax.grid(False)
            plt.xticks(rotation=90, ha='center')
            n_days = len(self.wdata["night"]["datetime"])
            axs[1].xaxis.set_major_locator(
                mdates.AutoDateLocator(minticks=n_days-1, maxticks=n_days+1)
            )

            if not toy:
                axs[0].set_title(self.loc_title)

            # T
            if not toy:
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
                s=200 if toy else 100 
            ) # nighttime air temperature
            axs[0].scatter(
                self.wdata["day"]["datetime"], self.wdata["day"]["SUR_TEMP"],
                edgecolors="black", clip_on=False,
                marker="v", color=time_to_color["day"], zorder=10, label="desorption\nconditions",
                s=200 if toy else 100 
            ) # daytime surface temperature
            # axs[0].set_title(self.location)
            axs[0].set_ylim(10, 65)
            axs[0].set_yticks([10 * _i for _i in range(1, 7)])
            axs[0].set_xlim(self.raw_data["datetime"].min(), self.raw_data["datetime"].max())

            # RH
            if not toy:
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
                marker="^", color=time_to_color["night"], zorder=10, s=200 if toy else 100,  label="capture conditions"
            ) # nighttime RH
            axs[1].scatter(
                self.wdata["day"]["datetime"], self.wdata["day"]["SUR_RH_HR_AVG"] / 100,
                edgecolors="black", clip_on=False,
                marker="v", color=time_to_color["day"], zorder=10, s=200 if toy else 100, label="release conditions"
            ) # day surface RH
            axs[1].set_yticks([0.2 * _i for _i in range(6)])
            if self.daynight_wdata.shape[0] > 1:
                axs[1].xaxis.set_major_formatter(my_date_format)
            if incl_legend:
                axs[1].legend(
                    prop={'size': 10}, ncol=1 if toy else 2, 
                    bbox_to_anchor=(0., 1.0 + legend_dy, 1.0 + legend_dx, .1), loc="center"
                )#, loc="center left")

            # already got legend above
            if save:
                plt.savefig(fig_dir + f"/weather_{self.loc_timespan_title}.pdf", format="pdf", bbox_inches="tight")

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

            # assert sequence day by day ie none missing
            days = self.daynight_wdata.loc[1:, "datetime"].dt.day.values

        def _gen_ads_des_conditions(self):
            self.ads_des_conditions = self.daynight_wdata.rename(columns=
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
def _(weather):
    weather.raw_data
    return


@app.cell
def _(weather):
    weather.raw_data.loc[weather.raw_data["datetime"].dt.month == 5]
    return


@app.cell
def _(Weather):
    weather = Weather([5, 6, 7, 8, 9], 2024, "Tucson")
    weather.raw_data
    return (weather,)


@app.cell
def _(weather):
    weather.daynight_wdata
    return


@app.cell
def _(weather):
    weather.ads_des_conditions
    return


@app.cell
def _(sns, weather):
    pp = sns.pairplot(
        weather.ads_des_conditions[1:], corner=True
    )
    pp.axes[1, 1].set_ylim(0, 1)
    pp.axes[3, 1].set_ylim(0, 1)
    pp.axes[3, 1].set_xlim(0, 1)
    pp.axes[3, 3].set_xlim(0, 1)

    T_range = [0, 65]
    assert T_range[1] > weather.ads_des_conditions[["ads T [°C]", "des T [°C]"]].max().max()
    assert T_range[0] < weather.ads_des_conditions[["ads T [°C]", "des T [°C]"]].min().min()

    pp.axes[0, 0].set_xlim(T_range[0], T_range[1])
    pp.axes[3, 2].set_xlim(T_range[0], T_range[1])
    pp.axes[2, 0].set_ylim(T_range[0], T_range[1])
    pp
    return


@app.cell
def _(weather):
    weather.viz_timeseries()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## dist'n of water deliveries
    """)
    return


@app.cell
def _(np):
    def score_fitness(wai, weather, alpha=10.0):
        # get dist'n of water dels
        water_dels = wai.water_del(weather.ads_des_conditions)
        # get worst-case water delivery, ignoring alpha % of hard cases.
        return np.percentile(water_dels, alpha)
    return (score_fitness,)


@app.cell
def _(score_fitness, wai, weather):
    fitness = score_fitness(wai, weather)
    return (fitness,)


@app.cell
def _(fitness, plt, wai, weather):
    plt.figure()
    plt.hist(wai.water_del(weather.ads_des_conditions))
    plt.axvline(fitness, color="C1")
    plt.ylabel("# days")
    plt.xlabel("water delivery")
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
    pop_size = 50

    wais = [
        WaterAdsorptionIsotherm(8, method="random step") 
        for _ in range(pop_size)
    ]

    fitnesses = np.array([score_fitness(wai, weather) for wai in wais])
    return fitnesses, pop_size, wais


@app.cell
def _(fitnesses, np):
    fitnesses[np.argsort(fitnesses)[-5:]]
    return


@app.cell
def _(fitnesses, np):
    np.argsort(fitnesses)[-5:]
    return


@app.cell
def _(fitnesses, plt):
    plt.figure()
    plt.xlabel("fitness")
    plt.ylabel("# soln's")
    plt.hist(fitnesses)
    plt.show()
    return


@app.cell
def _(WaterAdsorptionIsotherm, np, score_fitness, weather):
    def evolve(
        wais, n_elite=5, tourney_size=10, 
        n_rand=5, n_mutate=10, eps=0.05
    ):
        # what's the population size?
        pop_size = np.shape(wais)[0]

        # dim
        dim = wais[0].n

        # compute fitnesses of each individual
        fitnesses = np.array([score_fitness(wai, weather) for wai in wais])

        # grab elite individuals
        ids_elite = np.argsort(fitnesses)[-n_elite:]
        new_wais = [wais[i_elite] for i_elite in ids_elite]

        # tournament selection
        ids_pop = np.arange(pop_size - n_rand)
        for i in range(pop_size - n_elite - n_rand):
            # pick individuals for tournament
            ids_tourney = np.random.choice(
                ids_pop, size=tourney_size, replace=False
            )

            # select parents as the pair with highest fitness
            id_a = ids_tourney[np.argsort(fitnesses[ids_tourney])[-1]]
            id_b = ids_tourney[np.argsort(fitnesses[ids_tourney])[-2]]

            # mate
            wai = WaterAdsorptionIsotherm(dim, method="random step")
            alpha = np.random.rand()
            wai.bs = alpha * wais[id_a].bs + (1 - alpha) * wais[id_b].bs
            new_wais.append(wai)

        # random births for exploration
        for i in range(n_rand):
            new_wais.append(WaterAdsorptionIsotherm(dim, method="random step"))

        # mutation
        for i in range(n_mutate):
            # select non-elite individual to mutate
            id = np.random.choice(np.arange(n_elite, pop_size))

            # mutation
            delta_b = eps * np.sort(np.random.rand(dim + 1))

            new_wais[id].bs += delta_b
            new_wais[id].bs[new_wais[id].bs < 0.0] = 0.0
            new_wais[id].bs[new_wais[id].bs > 1.0] = 1.0

            # print("fitness A: ", fitnesses[id_a])
            # print("fitness B: ", fitnesses[id_b])

        return new_wais
    return (evolve,)


@app.cell
def _(evolve, fitnesses, np, plt, score_fitness, wais, weather):
    new_wais = evolve(wais, n_elite=5)
    new_fitnesses = np.array(
        [score_fitness(new_wai, weather) for new_wai in new_wais]
    )

    plt.figure()
    plt.xlabel("fitness")
    plt.ylabel("# soln's")
    plt.hist(fitnesses, alpha=0.5)
    plt.hist(new_fitnesses, alpha=0.5)
    plt.show()
    return


@app.cell
def _(
    WaterAdsorptionIsotherm,
    evolve,
    np,
    plt,
    pop_size,
    score_fitness,
    weather,
):
    def do_evolution(n_generations, pop_size, dim=15):
        # generate population
        wais = [
            WaterAdsorptionIsotherm(dim, method="random step") 
            for _ in range(pop_size)
        ]

        # fitness
        fitnesses = np.array([score_fitness(wai, weather) for wai in wais])

        fitness_progress = np.zeros(n_generations)
        fitness_progress[0] = np.max(fitnesses)

        for g in range(1, n_generations):
            wais = evolve(wais)
            fitnesses = np.array([score_fitness(wai, weather) for wai in wais])
            fitness_progress[g] = np.max(fitnesses)

        best_individual = wais[np.argmax(fitnesses)]

        return fitness_progress, wais, best_individual

    n_generations = 25
    dim = 25
    fitness_progress, evolved_wais, opt_wai = do_evolution(
        n_generations, pop_size, dim=dim
    )

    plt.figure()
    plt.plot(range(np.size(fitness_progress)), fitness_progress)
    plt.xlabel("iteration")
    plt.ylabel("fitness")
    plt.show()
    return (opt_wai,)


@app.cell
def _(opt_wai):
    opt_wai.draw()
    return


@app.cell
def _(colors, mpl, np, plt, score_fitness):
    def draw_opt(opt_wai, weather):
        p_over_p0s = np.linspace(0, 1, 100)

        # fig, axs = plt.subplots(
        #     2, 2, 
        #     gridspec_kw={'height_ratios': [1, 3], 'width_ratios': [2, 1]},
        #     figsize=(5, 7),
        #     layout="constrained"
        # )
        fig = plt.figure(figsize=(5, 5))
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
        axs[1, 0].set_ylabel("water adsorption")

        colormap = mpl.colormaps['coolwarm'] # or 'plasma', 'coolwarm', etc.
        norm = colors.Normalize(vmin=10.0, vmax=60.0)

        for T in np.linspace(10, 60, 4):
            axs[1, 0].plot(
                p_over_p0s, 
                [opt_wai.water_ads(T, p_over_p0) for p_over_p0 in p_over_p0s],
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
        axs[1, 0].set_ylim(0, 1)

        ###
        #   P/P0 distns
        ###
        p_over_p0_bins = np.linspace(0, 1, 25)
        axs[0, 0].hist(
            weather.ads_des_conditions["ads P/P0"], alpha=0.25, label="ads",
            bins=p_over_p0_bins
        )
        axs[0, 0].hist(
            weather.ads_des_conditions["des P/P0"], alpha=0.25, label="des", 
            bins=p_over_p0_bins
        )
        axs[0, 0].set_ylabel("# days")
        axs[0, 0].legend()

        ###
        #   working cap dist'n
        ###
        fitness = score_fitness(opt_wai, weather)
        plt.title(f"fitness = {fitness:.2f}")

        axs[1, 1].hist(
            opt_wai.water_del(weather.ads_des_conditions),
            orientation='horizontal', color="C4"
        )
        axs[1, 1].axhline(fitness, color="black", linestyle="--")
        axs[1, 1].set_xlabel("# days")
        axs[1, 1].set_ylabel("water delivery")

        plt.tight_layout()

        plt.show()
    return (draw_opt,)


@app.cell
def _(draw_opt, opt_wai, weather):
    draw_opt(opt_wai, weather)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## baseline: a stepped adsorption isotherm
    engineer for the average conditions
    """)
    return


@app.cell
def _(weather):
    avg_conditions = weather.ads_des_conditions.mean()
    avg_conditions
    return (avg_conditions,)


@app.cell
def _(avg_conditions):
    p_p0_step = (avg_conditions["ads P/P0"] + avg_conditions["des P/P0"]) / 2
    p_p0_step
    return (p_p0_step,)


@app.cell
def _(p_p0_step):
    p_p0_step * 100
    return


@app.cell
def _(WaterAdsorptionIsotherm, draw_opt, p_p0_step, weather):
    wai_manual = WaterAdsorptionIsotherm(
        100, method="manual step", n_step=int(p_p0_step * 100)
    )

    draw_opt(wai_manual, weather)
    return


@app.cell
def _(WaterAdsorptionIsotherm, draw_opt, np, score_fitness, weather):
    n_grid = 200
    wai_all_steps = [
        WaterAdsorptionIsotherm(n_grid, method="manual step", n_step=n_step) 
            for n_step in np.arange(2, n_grid-2)
    ]
    baseline_fitnesses = np.array(
        [score_fitness(wai, weather) for wai in wai_all_steps]
    )
    id_opt_baseline = np.argmax(baseline_fitnesses)

    draw_opt(wai_all_steps[id_opt_baseline], weather)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
