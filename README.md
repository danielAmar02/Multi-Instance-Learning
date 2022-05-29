# Multi Instances Learning in order to dectect Lymphocitose


In computer vision, a very common task is to use labelled data to classify images. There exists a lot of architectures and models that perform very well for this purpose. In some cases however, this approach may be inappropriate because of the nature of the problem we want to solve. In our problem, we have bags of images for each patient. The labelling is at the patient level. Then, we have no access to the labels of the instances (images of patients). This setting is known as Multi-Instance Learning.


Lymphocytosis is an increase in the number or proportion of lymphocytes in the blood. It can occur after an illness and be without risks, but it might represent something more serious, such as a blood cancer. The diagnosis is made on the basis of visual microscopic examination of the blood cells, together with the integration of clinical attributes (age, lymphocyte count...). Some clinical tests are required to validate the diagnosis, like flow cytometry.
The goal of this challenge is to provide a MIL framework which will predict with high accuracy if a new patient has Lymphocytosis. It would be very useful for two reasons :

• First, it automates the analysis of blood cells of the patients.

• Second, it is a good way to determine which patient should be referred for additional
analysis.
Using the bags of images of 163 patients together with some clinical attributes like age and lymphocyte count, we tried to propose a coherent approach to solve this challenge.

We reached 86% accuracy on the test dataset.

# Architecture 

After some research about Multi-Instance Learning, we decided to use the approach pro- posed in the paper [Attention-based Deep Multiple Instance Learning](https://arxiv.org/abs/1802.04712) which uses an attention mechanism.




## Using tabular data

To have an idea about the importance of the clinical data, we took a look to the correlation matrix. We can see that the variables ’YEAR’ (’DOB’) and ’LYMPH COUNT’ are highly correlated with the label of the patient. By curiosity, we performed a logistic regression with these two variables and obtained a highly score of 83% on the test dataset. This may sound surprising but we thought that this can be explained by the data in itself which does not represent the reality.
