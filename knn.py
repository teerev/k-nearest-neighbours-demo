import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from matplotlib.colors import ListedColormap
from sklearn import model_selection, neighbors
from matplotlib.ticker import FormatStrFormatter

style.use('ggplot')
pd.options.mode.chained_assignment = None

# User inputs
k_max = 501                        # max number of neighbours considered
k_min = 1                          # min number of neighbours considered
step = 20                          # step between k values
test_set_size = 0.7                # % of data held out for test set
grid_resolution = 100              # recommend maximum 500
n_0 = 1000                         # number of data points in class 0
n_1 = 1000                         # number of data points in class 1
mean_0 = [0.5, 3]                  # mean of class 0 data
cov_0 = [[1.5, 0.0], [0.0, 1.5]]   # covariance matrix of class 0 data
mean_1 = [0, 0]                    # mean of class 1 data
cov_1 = [[1.2, 0.0], [0.0, 1.2]]   # covariance matrix of class 1 data


def resolve_set_into_classes(x_set, y_set):
    
    """Takes the features data (2-column array) and corresponding response values 
    (1-column array of 0s and 1s) from the training or test set and as input and
    returns the same data resolved into subarrays for which responses equal 0 or 1."""
    
    # combine features and responses from a given set (training or testing set)
    x_y_set = pd.concat([x_set, y_set], axis=1)

    # filter by class
    x_y_set_1 = x_y_set[x_y_set['class'] == 1]
    x_y_set_0 = x_y_set[x_y_set['class'] == 0]

    # filter by features and (true) response values
    x_set_1 = np.squeeze(np.array(x_y_set_1.drop('class', axis=1)))
    x_set_0 = np.squeeze(np.array(x_y_set_0.drop('class', axis=1)))
    y_set_1 = np.squeeze(np.array(x_y_set_1.drop(['x1', 'x2'], axis=1)))
    y_set_0 = np.squeeze(np.array(x_y_set_0.drop(['x1', 'x2'], axis=1)))

    return x_set_1, x_set_0, y_set_1, y_set_0


def split_x_into_tp_fn_tn_fp(x_set_1, x_set_0, y_set_1, y_set_0, classifier):
    
    """Takes 4 input arrays as: features data (2 columns) and class values (1 column; 0s or 1s)
    from two distinct classes of data, and a classifier. Returns four distinct arrays 
    (2 columns each) of features data resolved into true/false positives/negatives"""
    
    y_set_1_pred = classifier.predict(x_set_1)
    y_set_0_pred = classifier.predict(x_set_0)

    tp_set = y_set_1_pred == y_set_1
    fn_set = y_set_1_pred != y_set_1
    tn_set = y_set_0_pred == y_set_0
    fp_set = y_set_0_pred != y_set_0

    x_set_tp = x_set_1[tp_set]
    x_set_fn = x_set_1[fn_set]
    x_set_tn = x_set_0[tn_set]
    x_set_fp = x_set_0[fp_set]

    return x_set_tp, x_set_fn, x_set_tn, x_set_fp


def plot_scatter_points(ax, x_tp, x_fn, x_tn, x_fp, dataset_label):
    
    """Plots true positives, true negatives, false positives and false negatives with distinct 
    scatter plot markers"""
    
    ax.scatter(x_tp[:, 0], x_tp[:, 1], s=50, marker='.', c='red',
               label='true positives ({0} set)'.format(dataset_label))

    ax.scatter(x_fn[:, 0], x_fn[:, 1], s=50, marker='x', c='red',
               label='false negatives ({0} set)'.format(dataset_label))

    ax.scatter(x_tn[:, 0], x_tn[:, 1], s=50, marker='.', c='blue',
               label='true negatives ({0} set)'.format(dataset_label))

    ax.scatter(x_fp[:, 0], x_fp[:, 1], s=50, marker='x', c='blue',
               label='false positives ({0} set)'.format(dataset_label))


# Draw data points from each Gaussian
x1_0, x2_0 = np.random.multivariate_normal(mean_0, cov_0, n_0).T
x1_1, x2_1 = np.random.multivariate_normal(mean_1, cov_1, n_1).T

# Join data points together into a single data frame
c1 = pd.DataFrame(np.transpose([x1_1, x2_1]), columns=['x1', 'x2'])
c0 = pd.DataFrame(np.transpose([x1_0, x2_0]), columns=['x1', 'x2'])
c1['class'] = 1
c0['class'] = 0
df = pd.concat([c1, c0])

# pick x2 and x1 as the features, and class (1 or 0) as the binary response
X = df[['x1', 'x2']]
y = df['class']

# Split data into training and test sets
X_trn, X_tst, y_trn, y_tst = model_selection.train_test_split(X, y, test_size=test_set_size)

# Further resolve the training and test sets into classes
X_trn_1, X_trn_0, y_trn_1, y_trn_0 = resolve_set_into_classes(X_trn, y_trn)
X_tst_1, X_tst_0, y_tst_1, y_tst_0 = resolve_set_into_classes(X_tst, y_tst)

# Form a regular 2D grid spanning the entire feature space to plot prediction regions
x1_grid = np.linspace(min(df['x1']), max(df['x1']), grid_resolution)
x2_grid = np.linspace(min(df['x2']), max(df['x2']), grid_resolution)
ee, nn = np.meshgrid(x1_grid, x2_grid)

# Flatten meshgrid to create two-column array of all grid coordinate pairs
prediction_grid = np.vstack([np.ravel(nn), np.ravel(ee)]).T
prediction_grid = np.flip(prediction_grid, axis=1)

# empty arrays to be populated in the loop for error vs. 1/k plot
one_over_k = np.empty(0)
test_error_rate = np.empty(0)
train_error_rate = np.empty(0)
sensitivity_tst = np.empty(0)
specificity_tst = np.empty(0)
sensitivity_trn = np.empty(0)
specificity_trn = np.empty(0)

# loop to produce successive images over k range
for k in range(k_max, k_min-1, -step):

    # Fit the training data
    clf = neighbors.KNeighborsClassifier(k)
    clf.fit(X_trn, y_trn)
    
    # Pass the coordinate pairs to the classifier to return class predictions for each point
    decision_regions = clf.predict(prediction_grid)

    # Repackage the class predictions into the shape corresponding to the meshgrid
    decision_regions = decision_regions.reshape(ee.shape)

    # classify training and test sets corresponding to entries of confusion matrix
    X_trn_tp, X_trn_fn, X_trn_tn, X_trn_fp = \
    split_x_into_tp_fn_tn_fp(X_trn_1, X_trn_0, y_trn_1, y_trn_0, clf)
    
    X_tst_tp, X_tst_fn, X_tst_tn, X_tst_fp = \
    split_x_into_tp_fn_tn_fp(X_tst_1, X_tst_0, y_tst_1, y_tst_0, clf)
    
    # count number of true positives, false positives, true negatives and false negatives
    tst_tp, tst_fn, tst_tn, tst_fp = len(X_tst_tp), len(X_tst_fn), len(X_tst_tn), len(X_tst_fp)
    trn_tp, trn_fn, trn_tn, trn_fp = len(X_trn_tp), len(X_trn_fn), len(X_trn_tn), len(X_trn_fp)

    # form arrays of error rates (updates with each kth loop iteration)
    test_error_rate = \
    np.append(test_error_rate, np.array([(tst_fn + tst_fp) / (tst_tp + tst_fn + tst_tn + tst_fp)]))
    
    train_error_rate = \
    np.append(train_error_rate, np.array([(trn_fn + trn_fp) / (trn_tp + trn_fn + trn_tn + trn_fp)]))

    # array of 1/k values for plotting test and training error
    one_over_k = np.append(one_over_k, np.array([1/k]))

    sensitivity_tst = np.append(sensitivity_tst, np.array([tst_tp / (tst_tp + tst_fn)]))
    specificity_tst = np.append(specificity_tst, np.array([tst_tn / (tst_tn + tst_fp)]))
    sensitivity_trn = np.append(sensitivity_trn, np.array([trn_tp / (trn_tp + trn_fn)]))
    specificity_trn = np.append(specificity_trn, np.array([trn_tn / (trn_tn + trn_fp)]))

    # Initialise plotting figure
    fig = plt.figure(figsize=(1920/100,  1080/100))
    custom_colors = ListedColormap(['#A1E1FF', '#FFC8C6'])

    ax1 = plt.subplot2grid((2, 5), (0,0), colspan=2, rowspan=2)
    ax2 = plt.subplot2grid((2, 5), (0,3), colspan=2, rowspan=2)
    ax3 = plt.subplot2grid((2, 5), (0,2), colspan=1, rowspan=1)
    ax4 = plt.subplot2grid((2, 5), (1,2), colspan=1, rowspan=1)
    
    # left panel (training data)
    ax1.set_xlabel(r'$x_1$')
    ax1.set_ylabel(r'$x_2$')
    ax1.set_xlim([min(df['x1']), max(df['x1'])])
    ax1.set_ylim([min(df['x2']), max(df['x2'])])

    # right panel (test data)
    ax2.set_xlabel(r'$x_1$')
    ax2.set_ylabel(r'$x_2$')
    ax2.set_xlim([min(df['x1']), max(df['x1'])])
    ax2.set_ylim([min(df['x2']), max(df['x2'])])
    
    # top-centre panel (error / accurary curve)
    ax3.set_xlabel(r'1/k')
    ax3.set_ylabel('error')
    ax3.set_xscale('log')
    ax3.set_xlim([1/k_max, 1])
    ax3.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax3.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax3.set_title('Error rates vs. 1/k')

    # bottom-centre panel (sensitivity / specificity plots)
    ax4.set_xlabel(r'1/k')
    ax4.set_ylabel('sensitivity / specificity')
    ax4.set_xscale('log')
    ax4.set_xlim([1/k_max, 1])
    ax4.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax4.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax4.set_title('Sens./Spec. vs. 1/k')

    # plot prediction regions
    ax1.pcolor(ee, nn,  decision_regions, cmap=custom_colors, shading='auto' )
    ax2.pcolor(ee, nn,  decision_regions, cmap=custom_colors, shading='auto' )

    # plot the training and test data scatter plots
    plot_scatter_points(ax1, X_trn_tp, X_trn_fn, X_trn_tn, X_trn_fp, dataset_label='training')
    ax1.set_title('{0}-nearest neighbours binary classifier: training set'.format(k), fontsize=14)

    plot_scatter_points(ax2, X_tst_tp, X_tst_fn, X_tst_tn, X_tst_fp, dataset_label='test')
    ax2.set_title('{0}-nearest neighbours binary classifier: test set'.format(k), fontsize=14)

    # plot test error array against 1/k array
    ax3.plot(one_over_k, test_error_rate, color='black', lw='2', label='test error')
    ax3.plot(one_over_k, train_error_rate, color='red', lw='2', label='training error')

    # plot sensitivity and specificity arrays against 1/k array
    ax4.plot(one_over_k, sensitivity_tst, c='black', lw='2', label='sensitivity (test set)')
    ax4.plot(one_over_k, specificity_tst, c='green', lw='2', label='specificity (test set)')
    ax4.plot(one_over_k, sensitivity_trn, c='red', lw='2', label='sensitivity (training set)')
    ax4.plot(one_over_k, specificity_trn, c='fuchsia', lw='2', label='specificity (training set)')

    plt.tight_layout()

    ax1.legend(loc='upper right')
    ax2.legend(loc='upper right')
    ax3.legend(loc='lower left')
    ax4.legend(loc='lower right')

    plt.savefig('{0}_knn.png'.format(k))

    # essential to avoid memory overflow
    plt.close(fig)
