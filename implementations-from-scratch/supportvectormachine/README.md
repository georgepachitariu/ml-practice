### Notes on SVM

#### Learning resources

1. [https://towardsdatascience.com/understanding-support-vector-machine-part-1-lagrange-multipliers-5c24a52ffc5e](https://towardsdatascience.com/understanding-support-vector-machine-part-1-lagrange-multipliers-5c24a52ffc5e)

2. [https://towardsdatascience.com/understanding-support-vector-machine-part-2-kernel-trick-mercers-theorem-e1e6848c6c4d](https://towardsdatascience.com/understanding-support-vector-machine-part-2-kernel-trick-mercers-theorem-e1e6848c6c4d)

3. [https://web.stanford.edu/~hastie/Papers/svmtalk.pdf](https://web.stanford.edu/~hastie/Papers/svmtalk.pdf)

The kernel trick: You change the input data (X), by applying a function to it.
Example: 2nd degree polinomial: You can get 2nd degree polinomial features by
having new_X = transpose(X) * X.

4. [https://www.reddit.com/r/MachineLearning/comments/3p9oqa/using_the_kernel_trick_on_a_logistic_regression/](https://www.reddit.com/r/MachineLearning/comments/3p9oqa/using_the_kernel_trick_on_a_logistic_regression/)

5. [https://stats.stackexchange.com/a/43806/258211](https://stats.stackexchange.com/a/43806/258211)

6. [https://www.quora.com/What-is-the-intuition-behind-Gaussian-kernel-in-SVM-How-can-I-visualize-the-transformation-function-%CF%95-that-corresponds-to-the-Gaussian-kernel-Why-is-the-Gaussian-kernel-popular](https://www.quora.com/What-is-the-intuition-behind-Gaussian-kernel-in-SVM-How-can-I-visualize-the-transformation-function-%CF%95-that-corresponds-to-the-Gaussian-kernel-Why-is-the-Gaussian-kernel-popular)

7. [https://stats.stackexchange.com/questions/37681/use-of-the-gamma-parameter-with-support-vector-machines/37713#37713](https://stats.stackexchange.com/questions/37681/use-of-the-gamma-parameter-with-support-vector-machines/37713#37713)

#### SVM versus Logistic Regression

1. Classification performance is almost identical in both cases.\[1]

2. KLR can provide class probabilities whereas SVM is a deterministic classifier. \[1]

3. KLR has a natural extension to multi-class classification 
whereas in SVM, there are multiple ways to extend it to multi-class 
classification. \[1]

4. Surprisingly or unsurprisingly, KLR also has optimal margin properties 
that the SVMs enjoy (well in the limit at least)! \[1]

5. KLR is computationally more expensive than SVM - O(N3) vs O(N2k) 
where k is the number of support vectors.
The classifier in SVM is designed such that it is defined only in terms 
of the support vectors, whereas in KLR, the classifier is defined over
 all the points and not just the support vectors. 
 This allows SVMs to enjoy some natural speed-ups 
 (in terms of efficient code-writing) that is hard to achieve for KLR. \[1]


#### Some personal notes

###### Small margin -> non-monotonic descent:
    
Since only the support vectors are taken into account at each iteration,
if the margin is small, then the number of sup. vector is small. 
This means that after each iteration, it can happen that new support vectors are selected, 
which increase the value of the loss function (normally the loss function should always decrease).


Sources:
\[1]  https://stats.stackexchange.com/questions/43996/kernel-logistic-regression-vs-svm

