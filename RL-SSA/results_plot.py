import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

kl_loss = pd.read_csv('/home/erskordi/Documents/UNM-files/Summer26/ME561/Project/2026-06-20/results_for_plotting/PPO_single_sat_env_bfee0_00000_0_2026-07-17_09-16-41.csv')

plt.plot(kl_loss.Step, kl_loss.Value)
plt.xlabel('Step')
plt.ylabel('KL Loss')
plt.title('KL Loss over Training Steps')
plt.grid()
# Save the plot as a PNG file
plt.savefig('/home/erskordi/Documents/UNM-files/Summer26/ME561/Project/2026-06-20/results_for_plotting/kl_loss_plot.png')
plt.show()

return_min = pd.read_csv('/home/erskordi/Documents/UNM-files/Summer26/ME561/Project/2026-06-20/results_for_plotting/PPO_single_sat_env_bfee0_00000_0_2026-07-17_09-16-41 (3).csv')
return_mean = pd.read_csv('/home/erskordi/Documents/UNM-files/Summer26/ME561/Project/2026-06-20/results_for_plotting/PPO_single_sat_env_bfee0_00000_0_2026-07-17_09-16-41 (2).csv')
return_max = pd.read_csv('/home/erskordi/Documents/UNM-files/Summer26/ME561/Project/2026-06-20/results_for_plotting/PPO_single_sat_env_bfee0_00000_0_2026-07-17_09-16-41 (1).csv')

plt.plot(return_mean.Step, return_mean.Value, label='Mean Return')
plt.fill_between(return_min.Step, return_min.Value, return_max.Value, color='lightblue', alpha=0.5, label='Return Range')
plt.xlabel('Step')
plt.ylabel('Return')
plt.title('Return over Training Steps')
plt.legend()
plt.grid()
# Save the plot as a PNG file
plt.savefig('/home/erskordi/Documents/UNM-files/Summer26/ME561/Project/2026-06-20/results_for_plotting/return_plot.png')
plt.show()