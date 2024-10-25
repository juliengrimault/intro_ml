import pdb
import numpy as np
import code_for_hw3_part2 as hw3

#-------------------------------------------------------------------------------
# Auto Data
#-------------------------------------------------------------------------------

# Returns a list of dictionaries.  Keys are the column names, including mpg.
auto_data_all = hw3.load_auto_data('auto-mpg.tsv')

# The choice of feature processing for each feature, mpg is always raw and
# does not need to be specified.  Other choices are hw3.standard and hw3.one_hot.
# 'name' is not numeric and would need a different encoding.
features = [('cylinders', hw3.raw),
            ('displacement', hw3.raw),
            ('horsepower', hw3.raw),
            ('weight', hw3.raw),
            ('acceleration', hw3.raw),
            ## Drop model_year by default
            ## ('model_year', hw3.raw),
            ('origin', hw3.raw)]

# Construct the standard data and label arrays
auto_data, auto_labels = hw3.auto_data_and_labels(auto_data_all, features)
print('auto data and labels shape', auto_data.shape, auto_labels.shape)

if False:                               # set to True to see histograms
    import matplotlib.pyplot as plt
    for feat in range(auto_data.shape[0]):
        print('Feature', feat, features[feat][0])
        # Plot histograms in one window, different colors
        plt.hist(auto_data[feat,auto_labels[0,:] > 0])
        plt.hist(auto_data[feat,auto_labels[0,:] < 0])
        plt.show()
        # Plot histograms in two windows, different colors
        fig,(a1,a2) = plt.subplots(nrows=2)
        a1.hist(auto_data[feat,auto_labels[0,:] > 0])
        a2.hist(auto_data[feat,auto_labels[0,:] < 0])
        plt.show()

#-------------------------------------------------------------------------------
# Analyze auto data
#-------------------------------------------------------------------------------

features2 = [('cylinders', hw3.one_hot),
            ('displacement', hw3.standard),
            ('horsepower', hw3.standard),
            ('weight', hw3.standard),
            ('acceleration', hw3.standard),
            ## Drop model_year by default
            ## ('model_year', hw3.raw),
            ('origin', hw3.one_hot)]

features3 = [('cylinders', hw3.one_hot),
            ('weight', hw3.standard),
            ]
# for (feature_name, f) in { 'all_raw': features, 'features2': features2, 'features3': features3 }.items():
#     data, labels = hw3.auto_data_and_labels(auto_data_all, f)
#     for T in [1, 10, 50]:
#         for alg_name, alg in { 'perceptron': hw3.perceptron, 'averaged_perceptron': hw3.averaged_perceptron }.items():
#             score = hw3.xval_learning_alg(alg, data, labels, k=10, T=T)
#             print("[{}, T={}, {}] score={}".format(feature_name, T, alg_name, score))

# data2, labels2 = hw3.auto_data_and_labels(auto_data_all, features2)
# th, th0 = hw3.averaged_perceptron(data, labels, {'T': 10})
# print(th, th0)

# 4.2 B) (Optional) Is there any set of two features you can use to attain comparable results as your best accuracy? What are they?
# features 3, use the 2 features from features2 with the highest coefficients, i.e cyclinders and weight

print('--------------------------------------------------------')

#-------------------------------------------------------------------------------
# Review Data
#-------------------------------------------------------------------------------

# Returns lists of dictionaries.  Keys are the column names, 'sentiment' and 'text'.
# The train data has 10,000 examples
review_data = hw3.load_review_data('reviews.tsv')

# Lists texts of reviews and list of labels (1 or -1)
review_texts, review_label_list = zip(*((sample['text'], sample['sentiment']) for sample in review_data))

# The dictionary of all the words for "bag of words"
dictionary = hw3.bag_of_words(review_texts)

# The standard data arrays for the bag of words
review_bow_data = hw3.extract_bow_feature_vectors(review_texts, dictionary)
review_labels = hw3.rv(review_label_list)
print('review_bow_data and labels shape', review_bow_data.shape, review_labels.shape)

#-------------------------------------------------------------------------------
# Analyze review data
#-------------------------------------------------------------------------------

print("")
print("")
print("")
print("----> Review Data")
print("------------------------------------------------")

# for T in [1, 10, 50]:
#     for alg_name, alg in { 'perceptron': hw3.perceptron, 'averaged_perceptron': hw3.averaged_perceptron }.items():
#         score = hw3.xval_learning_alg(alg, review_bow_data, review_labels, k=10, T=T)
#         print("[T={}, {}] score={}".format(T, alg_name, score))

# th, th0 = hw3.averaged_perceptron(review_bow_data, review_labels, {'T': 10})

def find_max_values(th, k=10):
    index_to_words_map = hw3.reverse_dict(dictionary)

    best = []
    worst = []
    n = np.shape(th)[0]

    for i in range(n):
        word = index_to_words_map[i]
        best.append((word, th[i, 0]))
        worst.append((word, th[i, 0]))
        if len(best) > k:
            best.sort(key=lambda x: x[1])
            del best[0]

        if len(worst) > k:
            worst.sort(key=lambda x: x[1])
            del worst[-1]
    
    best.sort(key=lambda x: x[1])
    worst.sort(key=lambda x: x[1])
    return (list(map(lambda x: x[0], best)), list(map(lambda x: x[0], worst)))

# print(find_max_values(th, k=10))
#-------------------------------------------------------------------------------
# MNIST Data
#-------------------------------------------------------------------------------

"""
Returns a dictionary formatted as follows:
{
    0: {
        "images": [(m by n image), (m by n image), ...],
        "labels": [0, 0, ..., 0]
    },
    1: {...},
    ...
    9
}
Where labels range from 0 to 9 and (m, n) images are represented
by arrays of floats from 0 to 1
"""
mnist_data_all = hw3.load_mnist_data(range(10))

print('mnist_data_all loaded. shape of single images is', mnist_data_all[0]["images"][0].shape)

# HINT: change the [0] and [1] if you want to access different images
def generate_data_labels(idx1, idx2):
    d0 = mnist_data_all[idx1]["images"]
    d1 = mnist_data_all[idx2]["images"]
    y0 = np.repeat(-1, len(d0)).reshape(1,-1)
    y1 = np.repeat(1, len(d1)).reshape(1,-1)

    # data goes into the feature computation functions
    data = np.vstack((d0, d1))
    # labels can directly go into the perceptron algorithm
    labels = np.vstack((y0.T, y1.T)).T

    return (data, labels)


def raw_mnist_features(x):
    """
    @param x (n_samples,m,n) array with values in (0,1)
    @return (m*n,n_samples) reshaped array where each entry is preserved
    """
    (n_samples,m,n) = np.shape(x)
    result = x.reshape(n_samples, m * n).T
    return result

def row_average_features(x):
    """
    This should either use or modify your code from the tutor questions.

    @param x (n_samples,m,n) array with values in (0,1)
    @return (m,n_samples) array where each entry is the average of a row
    """
    result = np.average(x, axis=2).T
    return result


def col_average_features(x):
    """
    This should either use or modify your code from the tutor questions.

    @param x (n_samples,m,n) array with values in (0,1)
    @return (n,n_samples) array where each entry is the average of a column
    """
    result = np.average(x, axis=1).T
    return result

def top_bottom_features(x):
    """
    This should either use or modify your code from the tutor questions.

    @param x (n_samples,m,n) array with values in (0,1)
    @return (2,n_samples) array where the first entry of each column is the average of the
    top half of the image = rows 0 to floor(m/2) [exclusive]
    and the second entry is the average of the bottom half of the image
    = rows floor(m/2) [inclusive] to m
    """
    n = np.shape(x)[0]
    mid = np.shape(x)[1] // 2
    top = np.average(x[:, :mid], axis=(1,2), keepdims=True).reshape(n ,1)
    bottom = np.average(x[:, mid:], axis=(1,2), keepdims=True).reshape(n ,1)
    result = np.concatenate([top, bottom], axis=1).T
    return result
    

#-------------------------------------------------------------------------------
# Analyze MNIST data
#-------------------------------------------------------------------------------

to_test = {
    '0 vs 1': generate_data_labels(0, 1),
    '2 vs 4': generate_data_labels(2, 4),
    '6 vs 8': generate_data_labels(6, 8),
    '9 vs 0': generate_data_labels(9, 0),
}

for (title, (data, labels)) in to_test.items():
    for (feature_method, feature) in [('row', row_average_features), ('col', col_average_features), ('top_bottom', top_bottom_features)]:
        processed_data = feature(data)
        acc = hw3.get_classification_accuracy(processed_data, labels)
        print("{}, {}: {}".format(title, feature_method, acc))
    
    print("---")


# 6.2F) (Optional) What does it mean if a binary classification accuracy is below 0.5, if your dataset is balanced (same number from each class)? Are these datasets balanced?
# it means it's random


