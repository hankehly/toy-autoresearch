"""Make a fake town's daily electricity use, Jan 2023 - Dec 2025.

We write the recipe ourselves, so we know the best score any model could get.
The agent gets Jan 2023 - Jun 2024. The judge scores Jul 2024 - Jun 2025, a year the agent
never sees. Jul - Dec 2025 is the vault, opened once at the end.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent  # the repo root (this file lives in scripts/)
rng = np.random.default_rng(7)

days = pd.date_range("2023-01-01", "2025-12-31", freq="D")
n = len(days)

# Temperature: Tokyo-like seasons (about 6 C in late Jan, 28 C in early Aug),
# plus warm and cold spells that last a few days.
season = 17 - 11 * np.cos(2 * np.pi * (days.dayofyear.to_numpy() - 25) / 365.25)
spell = np.zeros(n)
for i in range(1, n):
    spell[i] = 0.7 * spell[i - 1] + rng.normal(0, 1.8)
temp = season + spell
temp_forecast = temp + rng.normal(0, 1.0, n)  # the day-ahead forecast misses by about 1 C

# The secret recipe (MWh per day). Load follows the ACTUAL temperature,
# but a model only ever gets the FORECAST, so even a perfect model can't hit zero.
heating = 1.8 * np.maximum(0, 16 - temp)   # below 16 C, heaters come on
cooling = 2.5 * np.maximum(0, temp - 22)   # above 22 C, air conditioners come on
dow = days.dayofweek.to_numpy()
weekend = np.select([dow == 5, dow == 6], [-8.0, -12.0], 0.0)  # Saturday, Sunday
load = 100 + heating + cooling + weekend + rng.normal(0, 3, n)

df = pd.DataFrame({
    "date": days.strftime("%Y-%m-%d"),
    "temp_forecast": temp_forecast.round(1),
    "load": load.round(1),
})
(ROOT / "judge").mkdir(exist_ok=True)
(ROOT / "lab").mkdir(exist_ok=True)
df.to_csv(ROOT / "judge" / "full_data.csv", index=False)
df[df.date <= "2024-06-30"].to_csv(ROOT / "lab" / "data.csv", index=False)
print(f"Made {len(df)} days. The agent gets {(df.date <= '2024-06-30').sum()} of them (through Jun 2024).")
