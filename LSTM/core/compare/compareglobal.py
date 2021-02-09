
import covsirphy as cs
import pandas as pd
import datetime


def preparedataset():

    # Download datasets
    data_loader = cs.DataLoader("input")
    population_data = data_loader.population()
    jhu_data = data_loader.jhu()

    train = jhu_data.cleaned()
    countries = train["Country"].unique()
    total_data = []
    days_moving_average = 3

    for country in countries:
        try:
            s = cs.Scenario(jhu_data, population_data, country=country)
            diff = s.records_diff(variables=["Confirmed"],
                                  window=days_moving_average,
                                  show_figure=False)
            d = s.record_df
            
            # Add country name and number of new confirmed cases
            d["Country"] = country
            d["New Confirmed"] = diff.reset_index()["Confirmed"]
            d = d[:-3]
            total_data.append(d)
        except:
            print(country + " not found")
    
    train_df = pd.concat(total_data)

    train_global = train_df.groupby("Date").sum().reset_index()
    train_global.plot(x="Date", y="Infected")
    train_global.plot(x="Date", y="New Confirmed")
    # Prepare dataset
    train_global["Country"] = country
    train_global["Province"] = "-"
    train_global = train_global.drop(["Susceptible", "New Confirmed"], axis=1)
    return train_global


def compareglobal():
    # Download datasets
    data_loader = cs.DataLoader("input")
    population_data = data_loader.population()
    jhu_data = data_loader.jhu()
    country = "Japan" # Temporary

    train_global = preparedataset()

    models = [cs.SIR, cs.SIRF, cs.SIRD]
    dataframes = []

    for model in models:

        jhu_data = cs.JHUData.from_dataframe(train_global)
        s = cs.Scenario(jhu_data, population_data, country=country)
        s.add(days=35)
        s.trend(show_figure=False)

        TRAIN_UP_TO = pd.to_datetime('2020-10-01')
        summary = s.summary()
        summary["Start_dt"] = pd.to_datetime(summary["Start"], format="%d%b%Y")
        summary["End_dt"] = pd.to_datetime(summary["End"], format="%d%b%Y")
        query = summary[summary["End_dt"] > TRAIN_UP_TO]
        all_phases = query.index.tolist()
        s.combine(phases=all_phases)
        target_date = datetime.datetime.strftime(TRAIN_UP_TO - datetime.timedelta(days=1), format="%d%b%Y")
        s.separate(target_date)
        summary = s.summary()
        all_phases = summary.index.tolist()
        s.disable(phases=all_phases[:-1])
        s.enable(phases=all_phases[-1:])

        df = s.estimate(model=model)
        display(s.summary())
        dataframes.append(df)
    return df
    