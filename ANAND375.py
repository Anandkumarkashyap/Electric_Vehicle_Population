import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import probplot
from scipy.stats import f_oneway

plt.style.use('seaborn-v0_8') 
sns.set_theme()
print(plt.style.available)

df = pd.read_csv("C:\\Users\\Nishu\\Downloads\\Electric_Vehicle_Population_Data (2).csv")
print(f"Dataset shape: {df.shape}")

print("\nFirst 5 rows:")
print(df.head())


print("\nMissing values by column:")
print(df.isnull().sum())

categorical_cols = ['County', 'City', 'State', 'Electric Vehicle Type', 
                    'Clean Alternative Fuel Vehicle (CAFV) Eligibility', 'Electric Utility']
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

df['Electric Range'] = df['Electric Range'].fillna(df['Electric Range'].median())
df = df.dropna(subset=['Make', 'Model', 'Model Year'])
df.columns = df.columns.str.strip()

plt.figure(figsize=(10, 6))
ev_type_counts = df['Electric Vehicle Type'].value_counts()
ev_type_counts.plot(kind='bar', color=['#4C72B0', '#DD8452'])
plt.title('Distribution of Electric Vehicle Types', fontsize=14, fontweight='bold')
plt.xlabel('Vehicle Type', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


# Prepare data
top_makes = df['Make'].value_counts().nlargest(10)
top_makes_df = top_makes.reset_index()
top_makes_df.columns = ['Make', 'Count']

# Plot
plt.figure(figsize=(12, 6))
sns.barplot(data=top_makes_df, x='Make', y='Count', hue='Make', palette='viridis', legend=False)
plt.title('Top 10 Electric Vehicle Manufacturers', fontsize=14, fontweight='bold')
plt.xlabel('Manufacturer', fontsize=12)
plt.ylabel('Number of Vehicles', fontsize=12)
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


plt.figure(figsize=(12, 6))
yearly_counts = df['Model Year'].value_counts().sort_index()
sns.lineplot(x=yearly_counts.index, y=yearly_counts.values, marker='o', 
             linewidth=2.5, color='#4C72B0')
plt.title('Electric Vehicle Adoption Over Time', fontsize=14, fontweight='bold')
plt.xlabel('Model Year', fontsize=12)
plt.ylabel('Number of Vehicles', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Prepare data
yearly_counts = df['Model Year'].value_counts().sort_index()
yearly_df = pd.DataFrame({'Year': yearly_counts.index, 'Count': yearly_counts.values})

# Set plot style
sns.set_style("whitegrid")
plt.figure(figsize=(14, 7))

# Line plot
ax = sns.lineplot(data=yearly_df, x='Year', y='Count', marker='o', linewidth=2.5, color='#4C72B0')

# Gradient fill under the line
ax.fill_between(yearly_df['Year'], yearly_df['Count'], alpha=0.3, color='#4C72B0')

# Highlight peak point
max_year = yearly_df.loc[yearly_df['Count'].idxmax()]
plt.annotate(f"Peak: {int(max_year['Year'])}\n({max_year['Count']:,})",
             xy=(max_year['Year'], max_year['Count']),
             xytext=(max_year['Year'], max_year['Count'] + 500),
             ha='center',
             arrowprops=dict(arrowstyle='->', color='gray'))

# Add title and labels
plt.title('Electric Vehicle Adoption Over Time', fontsize=16, fontweight='bold')
plt.xlabel('Model Year', fontsize=13)
plt.ylabel('Number of Vehicles', fontsize=13)

# Format ticks
plt.xticks(yearly_df['Year'], rotation=45)
plt.grid(True, linestyle='--', alpha=0.6)

# Remove top and right spines
sns.despine()
plt.tight_layout()
plt.show()

range_by_type = df.groupby('Electric Vehicle Type')['Electric Range'].mean()
range_df = range_by_type.reset_index()
range_df.columns = ['Type', 'Range']


# Plot
sns.barplot(data=range_df, x='Type', y='Range', hue='Type', palette='rocket', legend=False)

plt.figure(figsize=(10, 6))
cafv_counts = df['Clean Alternative Fuel Vehicle (CAFV) Eligibility'].value_counts()
cafv_counts.plot(kind='pie', autopct='%1.1f%%', startangle=90, 
                 colors=['#4C72B0', '#DD8452', '#55A868'], 
                 explode=(0.05, 0.05, 0.05))

plt.title('CAFV Eligibility Distribution', fontsize=14, fontweight='bold')
plt.ylabel('')
plt.tight_layout()
plt.show()


top_counties = df['County'].value_counts().nlargest(10)  
top_counties_df = top_counties.reset_index()
top_counties_df.columns = ['County', 'Count']

# Set Seaborn style
sns.set_style("whitegrid")
plt.figure(figsize=(12, 8))

# Horizontal barplot
ax = sns.barplot(
    data=top_counties_df,
    x='Count',
    y='County',
    hue='County',
    palette='mako',
    legend=False
)

# Add value labels to bars
for container in ax.containers:
    ax.bar_label(container, fmt='%d', label_type='edge', padding=5, fontsize=10)

# Add titles and labels
plt.title('Top Counties by Electric Vehicle Count', fontsize=16, fontweight='bold')
plt.xlabel('Number of Vehicles', fontsize=13)
plt.ylabel('County', fontsize=13)

# Clean up the chart
plt.grid(axis='x', linestyle='--', alpha=0.6)
sns.despine(left=True, bottom=True)
plt.tight_layout()
plt.show()


import matplotlib.font_manager as fm
emoji_font = fm.FontProperties(fname="C:/Windows/Fonts/seguiemj.ttf")
plt.rcParams['font.family'] = [emoji_font.get_name()]

# Set Seaborn style
sns.set_style("whitegrid")
plt.figure(figsize=(16, 10))

# Clean column names in case of extra spaces
df.columns = df.columns.str.strip()

# Check and create 'price_data' safely
if 'Electric Vehicle Type' in df.columns and 'Base MSRP' in df.columns:
    price_data = df[['Electric Vehicle Type', 'Base MSRP']].dropna()
else:
    raise KeyError("Required columns 'Electric Vehicle Type' or 'Base MSRP' not found in the dataset.")


#  Convert to boxplot (fixes palette warning using `hue`)
ax = sns.boxplot(
    x='Electric Vehicle Type',
    y='Base MSRP',
    data=price_data,
    hue='Electric Vehicle Type',    
    palette='Set2',
    showfliers=False,
    linewidth=2
)

#  Remove legend (redundant)
if ax.legend_ is not None:
    ax.legend_.remove()


#  Add jittered stripplot (optional)
sns.stripplot(
    data=price_data,
    x='Electric Vehicle Type',
    y='Base MSRP',
    color='gray',
    alpha=0.4,
    jitter=True,
    size=3
)


#  Labels and titles
plt.title('Base MSRP by Electric Vehicle Type', fontsize=20, fontweight='bold')
plt.xlabel('Electric Vehicle Type', fontsize=15)
plt.ylabel('Base MSRP ($)', fontsize=15)
plt.xticks(rotation=20, fontsize=12)
plt.yticks(fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()

numeric_cols = df.select_dtypes(include=['float','int']).columns.tolist()
for col in numeric_cols:
    print('-' * 60)
    print(f"Column: {col}")
    print(f"Skewness: {df[col].skew():.2f}")
    print(f"Kurtosis: {df[col].kurt():.2f}")
    print('-' * 60)

fig, axes = plt.subplots(nrows=len(numeric_cols), ncols=2, figsize=(12, 6*len(numeric_cols)))

for i, col in enumerate(numeric_cols):
    # Histogram with KDE
    sns.histplot(df[col], kde=True, ax=axes[i,0], color='#4C72B0')
    axes[i,0].set_title(f'Distribution of {col}', fontsize=12)
    axes[i,0].set_xlabel(col)
    axes[i,0].set_ylabel('Frequency')
    axes[i,0].grid(True, linestyle='--', alpha=0.7)
    
    # Q-Q plot
    probplot(df[col], plot=axes[i,1])
    axes[i,1].set_title(f'Q-Q Plot of {col}', fontsize=12)
    axes[i,1].grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 8))
corr_matrix = df[numeric_cols].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
            fmt='.2f', linewidths=0.5, cbar_kws={'shrink': 0.8})
plt.title('Correlation Matrix of Numeric Features', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
sns.scatterplot(x='Model Year', y='Electric Range', data=df, 
                hue='Electric Vehicle Type', palette='Set1', alpha=0.7)
plt.title('Electric Range vs Model Year', fontsize=14, fontweight='bold')
plt.xlabel('Model Year', fontsize=12)
plt.ylabel('Electric Range (miles)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


top_makes_list = top_makes.index.tolist()

for make in top_makes_list[:3]:  # Analyze top 3 manufacturers
    make_data = df[df['Make'] == make]
    top_models = make_data['Model'].value_counts().nlargest(5)
    
    # Convert to DataFrame so we can use hue
    top_models_df = top_models.reset_index()
    top_models_df.columns = ['Model', 'Count']
    
    plt.figure(figsize=(10, 5))
    sns.set_style("whitegrid")
    
    # Add hue to apply palette safely
    ax = sns.barplot(
        data=top_models_df,
        x='Count',
        y='Model',
        hue='Model',
        palette="viridis",
        dodge=False,
        legend=False
    )

    # Optional: Add labels to bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%d', padding=5)

    plt.title(f'Top 5 {make} EV Models', fontsize=14, fontweight='bold')
    plt.xlabel('Number of Vehicles', fontsize=12)
    plt.ylabel('Model', fontsize=12)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


leg_dist_df = leg_dist_counts.reset_index()
leg_dist_df.columns = ['Legislative District', 'Count']
plt.figure(figsize=(14, 8))
sns.set_style("whitegrid")

sns.barplot(
    data=leg_dist_df,
    x='Legislative District',
    y='Count',
    hue='Legislative District',  
    palette='rocket',
    dodge=False,
    legend=False                  
)

plt.figure(figsize=(16, 10))
sns.set_style("whitegrid")

sns.boxplot(
    x='County',
    y='Electric Range',
    data=county_range_data,
    hue='County',
    palette='Set3',
    legend=False,
    linewidth=2,
    showfliers=False
)

plt.title('Electric Range by County', fontsize=18, fontweight='bold')
plt.xlabel('County', fontsize=14)
plt.ylabel('Electric Range (miles)', fontsize=14)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 8))
for county in top_counties_list[:5]:  # Plot top 5 counties
    county_data = df[df['County'] == county]
    yearly_count = county_data['Model Year'].value_counts().sort_index()
    sns.lineplot(x=yearly_count.index, y=yearly_count.values, 
                 label=county, linewidth=2.5)
    
plt.title('EV Adoption Over Time by Top Counties', fontsize=14, fontweight='bold')
plt.xlabel('Model Year', fontsize=12)
plt.ylabel('Number of Vehicles', fontsize=12)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

sns.barplot(x=top_utilities.values, y=top_utilities.index, palette="mako", legend=False)

top_ev_types = df['Electric Vehicle Type'].value_counts().nlargest(3).index
ev_subset = df[df['Electric Vehicle Type'].isin(top_ev_types)]

# Group data
groups = [group['Electric Range'].dropna() for name, group in ev_subset.groupby('Electric Vehicle Type')]

# Perform ANOVA
f_stat, p_val = f_oneway(*groups)

print("Hypothesis Test: ANOVA on Electric Range across EV Types")
print(f"F-statistic: {f_stat:.3f}, p-value: {p_val:.5f}")

if p_val < 0.05:
    print("Result: Reject the null hypothesis (Significant differences exist)")
else:
    print("Result: Fail to reject the null hypothesis (No significant difference)")
