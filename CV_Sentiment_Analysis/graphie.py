import matplotlib.pyplot as plt
import numpy as np

# Market Share Data
market_share_segments = [
    'Desktop OS',
    'Mobile OS',
    'Microsoft Office',
    'Cloud Productivity',
    'IaaS (Azure)',
    'PaaS (Azure)',
    'Xbox'
]

market_shares = [
    87.41,  # Desktop OS
    21.21,  # Mobile OS
    88.21,  # Microsoft Office
    53.4,   # Cloud Productivity
    29.4,   # IaaS (Azure)
    14.1,   # PaaS (Azure)
    27.55   # Xbox
]

# Revenue Data
years = ['2020', '2021', '2022', '2023']
revenue = [143.0, 168.1, 198.3, 242.1]  # in billions
growth_rate = [14, 17, 18, 22]  # percentage growth

# Create a figure with subplots
fig, axs = plt.subplots(2, 1, figsize=(10, 12))

# Market Share Pie Chart
axs[0].pie(market_shares, labels=market_share_segments, autopct='%1.1f%%', startangle=140)
axs[0].axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
axs[0].set_title('Microsoft Market Share by Segment')

# Revenue Bar Chart
axs[1].bar(years, revenue, color='skyblue', alpha=0.7, label='Revenue (in billions USD)')
axs[1].set_title('Microsoft Revenue Over Years')
axs[1].set_ylabel('Revenue (in billions USD)')
axs[1].set_ylim(0, 300)  # Set y-axis limit for better visibility

# Adding growth rate labels above the bars
for i, v in enumerate(revenue):
    axs[1].text(i, v + 5, f'{growth_rate[i]}%', ha='center', va='bottom', fontsize=10)

# Show grid
axs[1].grid(axis='y', linestyle='--', alpha=0.7)

# Show the plots
plt.tight_layout()
plt.show()
