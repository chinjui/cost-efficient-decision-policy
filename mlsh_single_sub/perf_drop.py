import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statistics
sns.set(style="darkgrid")

hidden8_256_file = "Ant-v1_hid8,256_ent1e-2_seed1179.txt"

combined_returns = []
macro_ratios = []
policy0_returns = []
policy1_returns = []
need_return = False

with open(hidden8_256_file) as f:
  for line in f:
    match = re.search('macro_acts: ([0-9\.]+)', line)
    if match is not None:
      macro_ratios.append(float(match.group(1)))
      need_return = True

    match = re.search('Episode .* return: ([0-9\.]+)', line)
    if match is not None:
      if need_return:
        combined_returns.append(float(match.group(1)))
        need_return = False

    match = re.search('sub 0: ([0-9\.]+), sub 1: ([0-9\.]+),', line)
    if match is not None:
      policy0_returns.append(float(match.group(1)))
      policy1_returns.append(float(match.group(2)))

macro_ratios = macro_ratios[:len(combined_returns)]
print("combined_returns;", len(combined_returns))
print("macro_ratios:", len(macro_ratios))
print("policy0_returns:", policy0_returns)
print("policy1_reutnrs:", policy1_returns)
policy0_returns = 3501.64
policy1_returns = 4020.30


# use the last 1000 results
macro_ratios = macro_ratios[-300:]
combined_returns = combined_returns[-300:]


macro_cost = 6.57
policy_costs = [1, 245.24]
costs = [(ratio * policy_costs[1] + (1-ratio) * policy_costs[0] + 0.2 * macro_cost) / policy_costs[1] for ratio in macro_ratios]

fig, axes = plt.subplots(ncols=2)

# plot 1: performance v.s. costs
relative_perf = [r / policy1_returns for r in combined_returns]
d = pd.DataFrame(data={'Perf (0 ~ large policy score)': relative_perf, 'Costs (%)': costs})
g = sns.regplot('Perf (0 ~ large policy score)', 'Costs (%)', data=d, color="m", ax=axes[0])
# axes[0].set_xlim([0, 1])
# axes[0].set_ylim([0, 1])

# plot 2: both %
relative_perf = [(r-policy0_returns) / (policy1_returns-policy0_returns) for r in combined_returns]
d = pd.DataFrame(data={'Perf (small policy ~ large policy score)': relative_perf, 'Costs (%)': costs})
g = sns.regplot('Perf (small policy ~ large policy score)', 'Costs (%)', data=d, color="m", ax=axes[1])
# plt.xlim(0, 1)
# plt.ylim(0, 1)
textstr = hidden8_256_file + '\n'
textstr = textstr + "small policy: %.2f, large policy: %.2f\n" % (policy0_returns, policy1_returns)
textstr += "costs: %.2f : %.2f" % (policy_costs[0], policy_costs[1])
fig.suptitle(textstr)

plt.show()
