from Imports import *
from Load_Data import num_cols
from Data_Cleaning import total_2


total_2['outcome'] = total_2['outcome'].map({'died': 0, 'euthanized': 1, 'lived': 2})

print(total_2['outcome'].unique())

sc = StandardScaler()
total_2[num_cols] = sc.fit_transform(total_2[num_cols])



pca = PCA(n_components=len(num_cols))

X_pca = pd.DataFrame(data=pca.fit_transform(total_2[num_cols]), columns=['PC'+str(i+1) for i in range(len(num_cols))])

var_exp = pd.Series(data=100*pca.explained_variance_ratio_, index=['PC'+str(i+1) for i in range(len(num_cols))])
print('Explained variance ratio per component:', round(var_exp,2), sep='\n')

print('Explained variance ratio with 6 components: '+str(round(var_exp.values[:6].sum(),5)))



# 10 PC's for Data visualization

pca6 = PCA(n_components=10)
X_pca6 = pd.DataFrame(data=pca6.fit_transform(total_2[num_cols]), columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6','PC7','PC8','PC9','PC10'])




# Create DataFrame for PCA loadings
pca_loadings = pd.DataFrame(data=pca6.components_, columns=num_cols)


# Create DataFrame for PCA loadings
pca_loadings = pd.DataFrame(data=pca6.components_, columns=num_cols)

fig, axs = plt.subplots(nrows=5, ncols=2, figsize=(24, 8))  # Adjust figsize if necessary
fig.suptitle('Loadings magnitude')

for i in range(5):  # Loop over rows
    for j in range(2):  # Loop over columns
        ax = axs[i][j]
        sns.barplot(ax=ax, x=pca_loadings.columns, y=pca_loadings.values[2 * i + j])
        ax.tick_params(axis='x', rotation=90)
        ax.title.set_text('PC' + str(2 * i + j + 1))

        # Set y-ticks every 0.05 units
        ax.set_yticks(np.arange(-0.5, 0.6, 0.1))  # adjust range if necessary based on the range of your data
        # Add grid to y-axis
        ax.yaxis.grid(True, linestyle='--', which='major', color='gray', alpha=.25)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout and leave space for the suptitle


BASE_DIR = r"C:\Users\vaque\PycharmProjects\Github\Horse-Survival\Logs"

def save_plot_to_base_dir(filename):
    full_path = os.path.join(BASE_DIR, filename)
    plt.savefig(full_path)
    print(f"Plot saved to {full_path}")


# 6 PC's for Data visualization


fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(24, 8))  # Adjust figsize if necessary
fig.suptitle('Loadings magnitude')
pca_loadings = pd.DataFrame(data=pca6.components_, columns=num_cols)

for i in range(2):  # Loop over rows
    for j in range(3):  # Loop over columns
        ax = axs[i][j]
        sns.barplot(ax=ax, x=pca_loadings.columns, y=pca_loadings.values[3 * i + j])
        ax.tick_params(axis='x', rotation=90)
        ax.title.set_text('PC' + str(3 * i + j + 1))

        # Set y-ticks every 0.05 units
        ax.set_yticks(np.arange(-0.5, 0.6, 0.1))  # adjust range if necessary based on the range of your data
        # Add grid to y-axis
        ax.yaxis.grid(True, linestyle='--', which='major', color='gray', alpha=.25)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])# Adjust layout and leave space for the suptitle



plt.title("6 PCs")
save_plot_to_base_dir("6 PCs.png")




# 3 PC's for Data visualization

# Create figure and axes
fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(24, 4))  # Adjust figsize if necessary
pca_loadings = pd.DataFrame(data=pca6.components_, columns=num_cols)

for i in range(3):  # Loop over the first three PCs
    ax = axs[i]
    sns.barplot(ax=ax, x=pca_loadings.columns, y=pca_loadings.values[i])
    ax.tick_params(axis='x', rotation=90)
    ax.title.set_text('PC' + str(i + 1))

    # Set y-ticks every 0.05 units
    ax.set_yticks(np.arange(-0.5, 0.6, 0.05))  # adjust range if necessary based on the range of your data
    # Add grid to y-axis
    ax.yaxis.grid(True, linestyle='--', which='major', color='gray', alpha=.25)

plt.tight_layout()


plt.title("3 PCs")
save_plot_to_base_dir("3 PCs.png")


# Visualize 3D PCA


X_pca6.rename(mapper={'PC1': 'PC1',
                      'PC2': 'PC2',
                      'PC3': 'PC3'}, axis=1, inplace=True)




# Define the color map
col = total_2['outcome'].map({0:'tab:red',1:'tab:blue',2:'tab:green'})



colors = ['tab:red','tab:blue','tab:green']
labelTups = [('Death', 'tab:red'),
             ('Euthanasied','tab:blue'),
             ('Live', 'tab:blue'),
             ]

# Plotting
fig = plt.figure(figsize=(9,6))
ax = fig.add_subplot(111, projection='3d')

lg = ax.scatter(X_pca6['PC1'],
                X_pca6['PC2'],
                X_pca6['PC3'],
                c=col)

ax.set_xlabel('$PC1$')
ax.set_ylabel('$PC2$')
ax.set_zlabel('$PC3$')
ax.title.set_text('Data including all failure outcomes')
ax.view_init(35, -10)

custom_lines = [plt.Line2D([],[], ls="", marker='.',
                           mec='k', mfc=c, mew=.1, ms=20) for c in colors]
ax.legend(custom_lines, [lt[0] for lt in labelTups],
          loc='center left', bbox_to_anchor=(1.0, .5))


plt.title("3D PCA")
save_plot_to_base_dir("3D PCA.png")

# Filter out the 'Live' data

mask = total_2['outcome'] != 2
X_pca6_filtered = X_pca6[mask]
col_filtered = col[mask]

# Define the new colors and labels after filtering out 'Live'
colors_filtered = ['tab:red', 'tab:blue']
labelTups_filtered = [('Death', 'tab:red'), ('Euthanized', 'tab:blue')]

# Plotting
fig = plt.figure(figsize=(9,6))
ax = fig.add_subplot(111, projection='3d')

lg = ax.scatter(X_pca6_filtered['PC1'],
                X_pca6_filtered['PC2'],
                X_pca6_filtered['PC3'],
                c=col_filtered)

ax.set_xlabel('$PC1$')
ax.set_ylabel('$PC2$')
ax.set_zlabel('$PC3$')
ax.title.set_text('Data not including Live')
ax.view_init(35, -10)

custom_lines = [plt.Line2D([], [], ls="", marker='.', mec='k', mfc=c, mew=.1, ms=20) for c in colors_filtered]
ax.legend(custom_lines, [lt[0] for lt in labelTups_filtered], loc='center left', bbox_to_anchor=(1.0, .5))



plt.title("3D PCA Filter out the Live")
save_plot_to_base_dir("3D PCA Filter out the Live.png")


#there's a significant overlap among the categories. This suggests that additional analysis or feature engineering
# might be required to more effectively distinguish among the three outcomes.

# Given that you need to retain 10 out of 11 PCs to capture 95% of the variance, PCA might not offer substantial reduction in this case.

plt.show()
