import numpy as np

X_train = np.genfromtxt("bird_sounds_features_train.csv", delimiter = ",")
y_train = np.genfromtxt("bird_sounds_labels_train.csv", delimiter = ",", dtype = int)
X_test = np.genfromtxt("bird_sounds_features_test.csv", delimiter = ",")
y_test = np.genfromtxt("bird_sounds_labels_test.csv", delimiter = ",", dtype = int)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)



# STEP 3
# assuming that there are K classes
# should return a numpy array with shape (K,)
def estimate_prior_probabilities(y):
    # your implementation starts below
    N = len(y)
    classes = [1,2,3]
    
    priors_list = []
    for c in classes:
        priors_list.append(np.sum(y == c) / N)
    class_priors = np.array(priors_list)
    # your implementation ends above
    return(class_priors)

class_priors = estimate_prior_probabilities(y_train)
print(class_priors)



# STEP 4
# assuming that there are K classes and D features
# should return two numpy arrays with shape (K, D)
def estimate_means_and_deviations(X, y):
    # your implementation starts below
    means = []
    deviations = []
    
    for c in [1, 2, 3]:
        X_c = X[y == c]  
        means.append(np.mean(X_c, axis=0))
        deviations.append(np.std(X_c, axis=0))
    
    means = np.array(means)
    deviations = np.array(deviations)
    # your implementation ends above
    return(means, deviations)

means, deviations = estimate_means_and_deviations(X_train, y_train)
print(means)
print(deviations)



# STEP 5
# assuming that there are N data points and K classes
# should return a numpy array with shape (N, K)
def calculate_score_values(X, means, deviations, class_priors):
    # your implementation starts below
    scores = []
    
    for c in range(len(class_priors)):
        m = means[c]
        d = deviations[c]
        prior = class_priors[c]
        
        prob = (1 / (np.sqrt(2 * np.pi) * d)) * np.exp(-((X - m) ** 2) / (2 * (d ** 2)))
        
        class_score = np.sum(np.log(prob), axis=1) + np.log(prior)
        
        scores.append(class_score)
        
    score_values = np.array(scores).T
    # your implementation ends above
    return(score_values)

scores_train = calculate_score_values(X_train, means, deviations, class_priors)
print(scores_train[0:10, :])

scores_test = calculate_score_values(X_test, means, deviations, class_priors)
print(scores_test[0:10, :])



# STEP 6
# assuming that there are K classes
# should return a numpy array with shape (K, K)
def calculate_confusion_matrix(y_truth, scores):
    # your implementation starts below
    classes = [1, 2, 3]
    N = len(classes)
    
    y_pred = []
    for s in scores:
        if s[0] >= s[1] and s[0] >= s[2]:
            y_pred.append(1)
        elif s[1] >= s[0] and s[1] >= s[2]:
            y_pred.append(2)
        else:
            y_pred.append(3)
            
    y_pred = np.array(y_pred)
    
    confusion_matrix = np.zeros((N, N)).astype(int)
    for r in range(N):
        for c in range(N):
            confusion_matrix[r, c] = np.sum((y_truth == classes[r]) & (y_pred == classes[c]))
    # your implementation ends above
    return(confusion_matrix)

confusion_train = calculate_confusion_matrix(y_train, scores_train)
print(confusion_train)
print("Training accuracy is {:.2f}%.".format(100 * np.sum(np.diag(confusion_train)) / np.sum(confusion_train)))

confusion_test = calculate_confusion_matrix(y_test, scores_test)
print(confusion_test)
print("Test accuracy is {:.2f}%.".format(100 * np.sum(np.diag(confusion_test)) / np.sum(confusion_test)))
