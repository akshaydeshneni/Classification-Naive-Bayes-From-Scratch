
# coding: utf-8

# In[1]:


import numpy as np
import scipy.io
import math
import geneNewData
import pandas as pd

def main():
    myID='7838'
    geneNewData.geneData(myID)
    Numpyfile0 = scipy.io.loadmat('digit0_stu_train'+myID+'.mat')
    Numpyfile1 = scipy.io.loadmat('digit1_stu_train'+myID+'.mat')
    Numpyfile2 = scipy.io.loadmat('digit0_testset'+'.mat')
    Numpyfile3 = scipy.io.loadmat('digit1_testset'+'.mat')
    train0_image = Numpyfile0.get('target_img')
    train1_image = Numpyfile1.get('target_img')
    test0_image = Numpyfile2.get('target_img')
    test1_image = Numpyfile3.get('target_img')

    #print([len(train0_image),len(train1_image),len(test0_image),len(test1_image)])
    #print('Your trainset and testset are generated successfully!')
    
    ### Reshaping Datasets
    
    train0_image_new = train0_image.reshape(*train0_image.shape[:1], -1)
    train1_image_new = train1_image.reshape(*train1_image.shape[:1], -1)

    test0_image_new = test0_image.reshape(*test0_image.shape[:1], -1)
    test1_image_new = test1_image.reshape(*test1_image.shape[:1], -1)
    
    #print(train0_image_new.shape)
    #print(test0_image_new.shape)
    #print(train1_image_new.shape)
    #print(test1_image_new.shape)

    ### Converting to a Pandas Dataframe
    
    pd_train0_image = pd.DataFrame(train0_image_new)
    pd_train1_image = pd.DataFrame(train1_image_new)
    pd_test0_image = pd.DataFrame(test0_image_new)
    pd_test1_image = pd.DataFrame(test1_image_new)
    
    #print(pd_train0_image)
    #print(pd_train1_image)
    #print(pd_test0_image)
    #print(pd_test1_image)

    ### Creating the Label Datasets
    
    pd_train0_label = pd.DataFrame(pd.np.empty((5000,1)))
    pd_train0_label[0] = 0
    #print(pd_train0_label)
    
    pd_train1_label = pd.DataFrame(pd.np.empty((5000,1)))
    pd_train1_label[0] = 1
    #print(pd_train1_label)

    pd_test0_label = pd.DataFrame(pd.np.empty((980,1)))
    pd_test0_label[0] = 0
    #print(pd_test0_label)
    
    pd_test1_label = pd.DataFrame(pd.np.empty((1135,1)))
    pd_test1_label[0] = 1
    #print(pd_test1_label)

    ### ---------------------------------------------------------------------------------------------------------------

    ### Calculating mu and sigma for the Digit 0 Training Set - Task 1

    pd_train0_features_extracted = pd.DataFrame()

    pd_train0_features_extracted['MEAN_BRIGHTNESS'] = pd_train0_image.mean(axis = 1)
    pd_train0_features_extracted['STD_DEV_BRIGHTNESS'] = pd_train0_image.std(axis = 1)

    # print(pd_train0_features_extracted)

    ### ---------------------------------------------------------------------------------------------------------------

    ### Task 1 - Calculating mu and sigma for the Digit 1 Training Set 

    pd_train1_features_extracted = pd.DataFrame()

    pd_train1_features_extracted['MEAN_BRIGHTNESS'] = pd_train1_image.mean(axis = 1)
    pd_train1_features_extracted['STD_DEV_BRIGHTNESS'] = pd_train1_image.std(axis = 1)

    # print(pd_train1_features_extracted)

    ### ---------------------------------------------------------------------------------------------------------------

    ### Task 2 - Calculating all Parameters for the Two-Class Naive Bayes Classifiers based upon the 
    ### 2-D data points generated above

    # Digit 0 Feature 1 Mean
    mean_feature1_digit0 = pd.DataFrame()
    mean_feature1_digit0['Mean_Feature1_Digit0'] = pd_train0_features_extracted['MEAN_BRIGHTNESS']
    mean_feature1_digit0 = mean_feature1_digit0.mean(axis = 0)

    print(mean_feature1_digit0)

    # Digit 0 Feature 1 Variance
    variance_feature1_digit0 = pd.DataFrame()
    variance_feature1_digit0['Variance_Feature1_Digit0'] = pd_train0_features_extracted['MEAN_BRIGHTNESS']
    variance_feature1_digit0 = variance_feature1_digit0.var(axis = 0)

    print(variance_feature1_digit0)

    # Digit 0 Feature 1 Standard Deviation
    std_feature1_digit0 = pd.DataFrame()
    std_feature1_digit0['StdDev_Feature1_Digit0'] = pd_train0_features_extracted['MEAN_BRIGHTNESS']
    std_feature1_digit0 = std_feature1_digit0.std(axis = 0)

    print(std_feature1_digit0)

    # Digit 0 Feature 2 Mean
    mean_feature2_digit0 = pd.DataFrame()
    mean_feature2_digit0['Mean_Feature2_Digit0'] = pd_train0_features_extracted['STD_DEV_BRIGHTNESS']
    mean_feature2_digit0 = mean_feature2_digit0.mean(axis = 0)

    print(mean_feature2_digit0)

    # Digit 0 Feature 2 Variance
    variance_feature2_digit0 = pd.DataFrame()
    variance_feature2_digit0['Variance_Feature2_Digit0'] = pd_train0_features_extracted['STD_DEV_BRIGHTNESS']
    variance_feature2_digit0 = variance_feature2_digit0.var(axis = 0)

    print(variance_feature2_digit0)

    # Digit 0 Feature 2 Standard Deviation
    std_feature2_digit0 = pd.DataFrame()
    std_feature2_digit0['StdDev_Feature2_Digit0'] = pd_train0_features_extracted['STD_DEV_BRIGHTNESS']
    std_feature2_digit0 = std_feature2_digit0.std(axis = 0)

    print(std_feature2_digit0)

    # Digit 1 Feature 1 Mean
    mean_feature1_digit1 = pd.DataFrame()
    mean_feature1_digit1['Mean_Feature1_Digit1'] = pd_train1_features_extracted['MEAN_BRIGHTNESS']
    mean_feature1_digit1 = mean_feature1_digit1.mean(axis = 0)

    print(mean_feature1_digit1)

    # Digit 1 Feature 1 Variance
    variance_feature1_digit1 = pd.DataFrame()
    variance_feature1_digit1['Variance_Feature1_Digit1'] = pd_train1_features_extracted['MEAN_BRIGHTNESS']
    variance_feature1_digit1 = variance_feature1_digit1.var(axis = 0)

    print(variance_feature1_digit1)

    # Digit 1 Feature 1 Standard Deviation
    std_feature1_digit1 = pd.DataFrame()
    std_feature1_digit1['StdDev_Feature1_Digit1'] = pd_train1_features_extracted['MEAN_BRIGHTNESS']
    std_feature1_digit1 = std_feature1_digit1.std(axis = 0)

    print(std_feature1_digit1)

    # Digit 1 Feature 2 Mean
    mean_feature2_digit1 = pd.DataFrame()
    mean_feature2_digit1['Mean_Feature2_Digit1'] = pd_train1_features_extracted['STD_DEV_BRIGHTNESS']
    mean_feature2_digit1 = mean_feature2_digit1.mean(axis = 0)

    print(mean_feature2_digit1)

    # Digit 1 Feature 2 Variance
    variance_feature2_digit1 = pd.DataFrame()
    variance_feature2_digit1['Variance_Feature2_Digit1'] = pd_train1_features_extracted['STD_DEV_BRIGHTNESS']
    variance_feature2_digit1 = variance_feature2_digit1.var(axis = 0)

    print(variance_feature2_digit1)

    # Digit 1 Feature 2 Standard Deviation
    std_feature2_digit1 = pd.DataFrame()
    std_feature2_digit1['StdDev_Feature2_Digit1'] = pd_train1_features_extracted['STD_DEV_BRIGHTNESS']
    std_feature2_digit1 = std_feature2_digit1.std(axis = 0)

    print(std_feature2_digit1)

    ### ---------------------------------------------------------------------------------------------------------------

    ### Task 3 - Calculating mu and sigma for the Digit 0 Testing Set

    pd_test0_features_extracted = pd.DataFrame()

    pd_test0_features_extracted['MEAN_BRIGHTNESS'] = pd_test0_image.mean(axis = 1)
    pd_test0_features_extracted['STD_DEV_BRIGHTNESS'] = pd_test0_image.std(axis = 1)

    # print(pd_test0_features_extracted)

    ### ---------------------------------------------------------------------------------------------------------------

    ### Task 3 - Calculating mu and sigma for the Digit 1 Testing Set 

    pd_test1_features_extracted = pd.DataFrame()

    pd_test1_features_extracted['MEAN_BRIGHTNESS'] = pd_test1_image.mean(axis = 1)
    pd_test1_features_extracted['STD_DEV_BRIGHTNESS'] = pd_test1_image.std(axis = 1)

    # print(pd_test1_features_extracted)

    ### ---------------------------------------------------------------------------------------------------------------

    ### Task 3 - Naive Bayes Classifier is trained with Training Data for both Digits 0 and 1

    # Final Training Sets X_train and y_train

    X_train_digit0 = pd.DataFrame()
    X_train_digit1 = pd.DataFrame()

    X_train_digit0 = pd_train0_features_extracted
    X_train_digit1 = pd_train1_features_extracted

    # print(X_train_digit0)
    # print(X_train_digit1)

    y_train_digit0 = pd.DataFrame()
    y_train_digit1 = pd.DataFrame()

    y_train_digit0 = pd_train0_label
    y_train_digit1 = pd_train1_label

    # print(y_train_digit0)
    # print(y_train_digit1)

    # Final Testing Sets X_test and y_test

    X_test = pd.DataFrame()
    X_test = pd_test0_features_extracted
    X_test = X_test.append(pd_test1_features_extracted)
    X_test = X_test.reset_index(drop = True)

    y_test = pd.DataFrame()
    y_test = pd_test0_label
    y_test = y_test.append(pd_test1_label)
    y_test = y_test.reset_index(drop = True)

    # print(X_test)
    # print(y_test)

    def gaussian_digit0(Dataset):

        probability_X_given_Y_digit0 = []

        for index, row in Dataset.iterrows():

                exponent1 = (np.exp(-np.power(Dataset.iloc[index, 0] - mean_feature1_digit0.iloc[0], 2.) / (2 * np.power(std_feature1_digit0.iloc[0], 2.)))) 
                exponent2 = (np.exp(-np.power(Dataset.iloc[index, 1] - mean_feature2_digit0.iloc[0], 2.) / (2 * np.power(std_feature2_digit0.iloc[0], 2.))))

                result = ((exponent1 / (np.sqrt(2 * np.pi) * std_feature1_digit0.iloc[0]))) * ((exponent2 / (np.sqrt(2 * np.pi) * std_feature2_digit0.iloc[0])))

                probability_X_given_Y_digit0.append(result)

        return probability_X_given_Y_digit0



    def gaussian_digit1(Dataset):

        probability_X_given_Y_digit1 = []

        for index, row in Dataset.iterrows():

                exponent1 = (np.exp(-np.power(Dataset.iloc[index, 0] - mean_feature1_digit1.iloc[0], 2.) / (2 * np.power(std_feature1_digit1.iloc[0], 2.)))) 
                exponent2 = (np.exp(-np.power(Dataset.iloc[index, 1] - mean_feature2_digit1.iloc[0], 2.) / (2 * np.power(std_feature2_digit1.iloc[0], 2.))))

                result = ((exponent1 / (np.sqrt(2 * np.pi) * std_feature1_digit1.iloc[0]))) * ((exponent2 / (np.sqrt(2 * np.pi) * std_feature2_digit1.iloc[0])))

                probability_X_given_Y_digit1.append(result)

        return probability_X_given_Y_digit1



    train_prob_X_given_Y_digit0 = gaussian_digit0(X_train_digit0)
    train_prob_X_given_Y_digit1 = gaussian_digit1(X_train_digit1)

    pd_train_prob_X_given_Y_digit0 = pd.DataFrame(train_prob_X_given_Y_digit0)
    pd_train_prob_X_given_Y_digit1 = pd.DataFrame(train_prob_X_given_Y_digit1)

    # print(pd_train_prob_X_given_Y_digit0)
    # print(pd_train_prob_X_given_Y_digit1)

    def predict_probability(Dataset):

        pred_y = []

        for index, row in Dataset.iterrows():

            result = 0.5 * (Dataset.iloc[index, 0])

            pred_y.append(result)

        return pred_y

    def final_prediction(Dataset1, Dataset2):

        final_pred_y = []

        for index in range(2115):
        
            if Dataset1.iloc[index, 0] > Dataset2.iloc[index, 0]:
                result = 0
            else: result = 1
        
            final_pred_y.append(result)

        return final_pred_y

    def model(Dataset):

        test_prob_X_given_Y_gaussian0 = gaussian_digit0(Dataset)
        pd_test_prob_X_given_Y_gaussian0 = pd.DataFrame(test_prob_X_given_Y_gaussian0)
        test_prob_X_given_Y_gaussian1 = gaussian_digit1(Dataset)
        pd_test_prob_X_given_Y_gaussian1 = pd.DataFrame(test_prob_X_given_Y_gaussian1)

    #     print(pd_test_prob_X_given_Y_gaussian0)
    #     print(pd_test_prob_X_given_Y_gaussian1)

        pred_prob_y_gaussian0 = predict_probability(pd_test_prob_X_given_Y_gaussian0)
        pd_pred_prob_y_gaussian0 = pd.DataFrame(pred_prob_y_gaussian0)
        pred_prob_y_gaussian1 = predict_probability(pd_test_prob_X_given_Y_gaussian1)
        pd_pred_prob_y_gaussian1 = pd.DataFrame(pred_prob_y_gaussian1)

    #     print(pd_pred_prob_y_gaussian0)
    #     print(pd_pred_prob_y_gaussian1)

        final_pred_y = final_prediction(pd_pred_prob_y_gaussian0, pd_pred_prob_y_gaussian1)

    #     print(final_pred_y)
        return final_pred_y

    test_pred_y = model(X_test)
    pd_test_pred_y = pd.DataFrame(test_pred_y)

    # print(pd_test_pred_y)

    pd_test_pred_y_digit0 = pd_test_pred_y.iloc[0:980]
    # print(pd_test_pred_y_digit0)
    pd_test_pred_y_digit1 = pd_test_pred_y.iloc[980:2115]
    pd_test_pred_y_digit1 = pd_test_pred_y_digit1.reset_index(drop = True)
    # print(pd_test_pred_y_digit1)
    y_test_digit0 = y_test.iloc[0:980]
    # print(y_test_digit0)
    y_test_digit1 = y_test.iloc[980:2115]
    # print(y_test_digit1)

    # pd_test_pred_y_digit0[0].isin(y_test_digit0[0]).value_counts()




    def validation(Predicted, Label):

        validation_arr = []

        for index, row in Predicted.iterrows():

            if(Predicted.iloc[index, 0] == Label.iloc[index, 0]):
                result = "Match"
            else: result = "NoMatch"

            validation_arr.append(result)
        return validation_arr



    validation_array_digit0 = validation(pd_test_pred_y_digit0, y_test_digit0)
    validation_array_digit1 = validation(pd_test_pred_y_digit1, y_test_digit1)
    pd_validation_digit0 = pd.DataFrame(validation_array_digit0)
    pd_validation_digit1 = pd.DataFrame(validation_array_digit1)

    # print(validation_array_digit0)


    # print(pd_validation_digit0)
    # print(pd_validation_digit1)


    def accuracy(ValidationArray):

        match_occurance = np.count_nonzero(ValidationArray == "Match")

        result = (match_occurance / len(ValidationArray))

        return result

    accuracy_test_digit0 = accuracy(pd_validation_digit0)
    print("Accuracy of Test Set Digit 0 : ", accuracy_test_digit0) 

    accuracy_test_digit1 = accuracy(pd_validation_digit1)
    print("Accuracy of Test Set Digit 1 : ", accuracy_test_digit1) 
    
    
    
    
if __name__ == '__main__':
    main()

