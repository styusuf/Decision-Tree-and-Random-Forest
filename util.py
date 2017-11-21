import numpy as np



# This method computes entropy for information gain
def entropy(class_y):
    # Input:            
    #   class_y         : list of class labels (0's and 1's)
    
    # Compute the entropy for a list of classes
    #
    # Example:
    #    entropy([0,0,0,1,1,1,1,1,1]) = 0.92

    # Favorite entropy and information gain explanation/formulas: https://stackoverflow.com/questions/1859554/what-is-entropy-and-information-gain
    count = float(len(class_y))

    class_0 = []
    class_1 = []
    
    

    for x in class_y:
        if int(x) == 0:
            class_0.append(x)
        elif int(x) == 1:
            class_1.append(x)
    
    if count != 0:
        p_0 = len(class_0)/count
        p_1 = len(class_1)/count
    else:
        p_0 = 0.0
        p_1 = 0.0
    
    entropy = 0
    
    if p_0 == 0 and p_1 != 0:
        entropy = abs(p_1*np.log2(p_1))
        
    elif p_0 != 0 and p_1 == 0:
        entropy = abs(p_0*np.log2(p_0))
    
    elif p_0 != 0 and p_1 != 0:
        entropy = -(p_0*np.log2(p_0)) - p_1*np.log2(p_1)
        
    return entropy


def partition_classes(X, y, split_attribute, split_val):
    # Inputs:
    #   X               : data containing all attributes
    #   y               : labels
    #   split_attribute : column index of the attribute to split on
    #   split_val       : either a numerical or categorical value to divide the split_attribute
    
    # Partition the data(X) and labels(y) based on the split value - BINARY SPLIT.
    #   
    #
    # You can perform the partition in the following way
    # Numeric Split Attribute:
    #   Split the data X into two lists(X_left and X_right) where the first list has all
    #   the rows where the split attribute is less than or equal to the split value (the mean), and the 
    #   second list has all the rows where the split attribute is greater than the split 
    #   value. Also create two lists(y_left and y_right) with the corresponding y labels.
    #
    # Categorical Split Attribute:
    #   Split the data X into two lists(X_left and X_right) where the first list has all 
    #   the rows where the split attribute is equal to the split value, and the second list
    #   has all the rows where the split attribute is not equal to the split value.
    #   Also create two lists(y_left and y_right) with the corresponding y labels.

    '''
    Example:
    
    X = [[3, 'aa', 10],                 y = [1,
         [1, 'bb', 22],                      1,
         [2, 'cc', 28],                      0,
         [5, 'bb', 32],                      0,
         [4, 'cc', 32]]                      1]
    
    Here, columns 0 and 2 represent numeric attributes, while column 1 is a categorical attribute.
    
    Consider the case where we call the function with split_attribute = 0 and split_val = 3 (mean of column 0)
    Then we divide X into two lists - X_left, where column 0 is <= 3  and X_right, where column 0 is > 3.
    
    X_left = [[3, 'aa', 10],                 y_left = [1,
              [1, 'bb', 22],                           1,
              [2, 'cc', 28]]                           0]
              
    X_right = [[5, 'bb', 32],                y_right = [0,
               [4, 'cc', 32]]                           1]

    Consider another case where we call the function with split_attribute = 1 and split_val = 'bb'
    Then we divide X into two lists, one where column 1 is 'bb', and the other where it is not 'bb'.
        
    X_left = [[1, 'bb', 22],                 y_left = [1,
              [5, 'bb', 32]]                           0]
              
    X_right = [[3, 'aa', 10],                y_right = [1,
               [2, 'cc', 28],                           0,
               [4, 'cc', 32]]                           1]
               
    ''' 
    


    cat = False

    X_left = []
    X_right = []
    
    y_left = []
    y_right = []
    
    X = np.array(X)
    
    try:
        float(split_val)
    except:
        cat = True
        
    if cat is True:
        for i in range(len(X)):
            if X[i,split_attribute] == split_val:
                X_left.append(X[i])
                y_left.append(y[i])
            else:
                X_right.append(X[i])
                y_right.append(y[i])
                
    if cat is False:
        for i in range(len(X)):
            if X[i,split_attribute].astype(float) <= split_val:
                X_left.append(X[i])
                y_left.append(y[i])
            else:
                X_right.append(X[i])
                y_right.append(y[i])
                
    return X_left, X_right, y_left, y_right

    
def information_gain(previous_y, current_y):
    # Inputs:
    #   previous_y: the distribution of original labels (0's and 1's)
    #   current_y:  the distribution of labels after splitting based on a particular
    #               split attribute and split value
    
    # Compute and return the information gain from partitioning the previous_y labels
    # into the current_y labels.

    # Use the entropy function above to compute information gain.

    
    """
    Example:
    
    previous_y = [0,0,0,1,1,1]
    current_y = [[0,0], [1,1,1,0]]
    
    info_gain = 0.45915
    """
    info_gain = 0

    H = entropy(previous_y)
    left = current_y[0]
    right = current_y[1]

    total_len = float(len(left) + len(right))

    H_L = entropy(left)
    H_R = entropy(right)

    P_L = len(left)/total_len
    P_R = len(right)/total_len

    info_gain = H - ((H_L*P_L) + (H_R*P_R))
    

    return info_gain

ent = entropy([0,0,0,1,1,1,1,1,1])
print("Entropy:")
print(ent)

X = [[3, 'aa', 10], [1, 'bb', 22], [2, 'cc', 28], [5, 'bb', 32], [4, 'cc', 32]]
y = [1, 1, 0, 0, 1]

X_left, X_right, y_left, y_right = partition_classes(X,y, 0, 3)
print("X Left: ")
print(X_left)
print("X Right: ")
print(X_right)
print("y Left: ")
print(y_left)
print("y Right: ")
print(y_right)


previous_y = [0,0,0,1,1,1]
current_y = [[0,0], [1,1,1,0]]

info_gain = information_gain(previous_y, current_y)
print("Information gain: ")
print(info_gain)




