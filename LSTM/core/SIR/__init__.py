import covsirphy as cs
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
sns.set()

class SIR():

    def __init__(self, COUNTRY, TRAIN_UP_TO):
        self.country     = COUNTRY
        self.TRAIN_UP_TO = TRAIN_UP_TO
        self.snl         = None

    def init_data(self):

        # Download datasets
        data_loader = cs.DataLoader("input")
        jhu_data = data_loader.jhu()
        population_data = data_loader.population()

        # total_data = []
        # days_moving_average = 1

        # s = cs.Scenario(jhu_data, population_data, country=self.country)
        # s.complement()
        # diff = s.records_diff(variables=["Confirmed"],
        #                       window=days_moving_average, show_figure=False)
        # d = s.record_df

        # # Add self.country name and number of new confirmed cases
        # d["Country"] = self.country
        # d["New Confirmed"] = diff.reset_index()["Confirmed"]
        # d = d[:-days_moving_average]
        # total_data.append(d)
                
        # train_df = pd.concat(total_data)
        # train_df.head()

        # pivot_date = "'2020-10-01'"
        # train_section = train_df[train_df["Country"] == self.country].query("Date<=" + pivot_date)
        # train_section.plot(x="Date", y="New Confirmed")
        # train_section["Province"] = "-"
        # jhu_data = cs.JHUData.from_dataframe(train_section)

        self.snl = cs.Scenario(jhu_data, population_data, country=self.country)
        self.snl.add(days=35)
        self.snl.trend(show_figure=False)
        return self.snl

    def estimate(self):
        # Parameter estimation of SIR model
        self.snl.add(days=35)
        self.snl.estimate(cs.SIRF, show_figure=False,
                                    auto_complement=False)
        return self.snl

    def simulate(self):
        self.snl.add(days=35)
        return self.snl.simulate()

    def plot(self, pred, predDate, showTrainData):
        self.COUNTRY = "Netherlands"
        self.TRAIN_UP_TO = pd.to_datetime('2020-10-01')

        fig, ax = plt.subplots(1, 1, figsize = (9, 4))
        ax.tick_params(axis='x', rotation=45)
        fig.suptitle(self.COUNTRY + ' confirmed cases prediction')
        sns.lineplot(y = pred, x = predDate, ax = ax, linewidth=4.5)
        sns.lineplot(y = 'New Confirmed', x = 'Date', data = showTrainData, ax = ax, linewidth=4.5)
        ax.legend(['Pred', 'Train'])
        ax.axvline(x=self.TRAIN_UP_TO, ymin = 0.0, ymax = 1.0, linestyle='--', lw = 1, color = '#808080')
        ax.grid(True)
        plt.show()