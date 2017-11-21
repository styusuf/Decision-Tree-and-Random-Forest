from util import entropy, information_gain, partition_classes
import numpy as np

class DecisionTree(object):
    def __init__(self):
        # Initializing the tree as an empty dictionary or list, as preferred
        #self.tree = []
        self.tree = {}
        pass

    def learn(self, X, y):
        # Train the decision tree (self.tree) using the the sample X and labels y
        # We will make use of the functions in utils.py to train the tree
        
        def build_tree(X,y):
            # This function will build the trees for the model. It's a recursive funtion that will stop when there is
            # only one unique value in y (the label), meaning we have a pure leaf.
        
            if len(set(y)) == 1:  # checks if we have a pure leaf
                return {'label': int(y[0])}
            
            else:
                X_np = np.array(X)
                n_factors = len(X_np[0])
                info_gain = {} # Will hold the information we get from splitting on each feature/column
                cats_dict = {} # will hold the information gain from each categorical column
                for col_index in range(n_factors):
                    category = False
                    try:
                        float(X_np[:,col_index][0])
                    except:
                        category = True
                    
                    # if column numerical, split using the column mean and calculate information gain
                    if category is False: 
                        vals = X_np[:,col_index].astype(float)
                        mean = np.mean(vals)
                        X_left, X_right, y_left, y_right = partition_classes(X, y, col_index, mean)
                        info_gain[col_index] = information_gain(y, [y_left, y_right])
                    
                    # if column is categorical, calculate info gain from splitting on each of the column values.
                    # Pick value with largest information gain to be the split value for the column
                    elif category is True:
                        vals = X_np[:,col_index]
                        cat_info_gain = {}
                        for val in set(vals):
                            X_left, X_right, y_left, y_right = partition_classes(X, y, col_index, val)
                            cat_info_gain[val] = information_gain(y, [y_left, y_right])
                        max_cat = max(cat_info_gain, key=cat_info_gain.get)
                        cats_dict[col_index] = max_cat
                        info_gain[col_index] = cat_info_gain[max_cat]
                            
                winner_col = max(info_gain, key=info_gain.get) # Get the col index with the largest information gain. We will split on that column
                
                cat_winner = False
                
                try:
                    float(X_np[:,winner_col][0])
                except:
                    cat_winner = True
                
                if cat_winner is False:
                    win_vals = X_np[:,winner_col].astype(float)
                    win_mean = np.mean(win_vals)
                    Xleft, Xright, yleft, yright = partition_classes(X, y, winner_col, win_mean)
                    # Split on the winner column, then call the build_tree funtion for left and right split
                    return {'splitval': win_mean, 'splitcol': winner_col, 'left': build_tree(Xleft, yleft), 'right': build_tree(Xright, yright)}
                    
                elif cat_winner is True:
                    max_cat = cats_dict[winner_col]
                    Xleft, Xright, yleft, yright = partition_classes(X, y, winner_col, max_cat)
                    # Split on the winner column, then call the build_tree funtion for left and right split
                    return {'splitval': max_cat, 'splitcol': winner_col, 'left': build_tree(Xleft, yleft), 'right': build_tree(Xright, yright)}
            
        self.tree = build_tree(X, y) # Commence tree building
                        
    def classify(self, record):
        # This function will predict the classification of a record given the decision tree created
        # Classify the record using self.tree and return the predicted label
        
        def predict(tree, record):
            #This is a recursive function that navigates through a tree to find the classification of a new record.
            
            assert isinstance(tree, dict)
            
            if tree.has_key('label') is True:
                val = tree['label']
                return val
            
            else:
                col_index = tree['splitcol']
                split_val = tree['splitval']
                cat_var = False # Check if the attr we are splitting on is categorical
                try:
                    float(split_val)
                except:
                    cat_var = True
                    
                if cat_var is True:
                    if record[col_index] == split_val:
                        return predict(tree['left'], record)
                    else:
                        return predict(tree['right'], record)
                else:
                    if record[col_index] <= split_val:
                        return predict(tree['left'], record)
                    else:
                        return predict(tree['right'], record)
            
        
        prediction = predict(self.tree, record)
        
        return prediction

#Example:
X = [[3, 'aa', 10], [1, 'bb', 22], [2, 'cc', 28], [5, 'bb', 32], [4, 'cc', 32]]
y = [1, 1, 0, 0, 1]
dt = DecisionTree()
dt.learn(X,y)
display(dt.tree)
record = [5, 'cc', 30]
print(dt.classify(record))



