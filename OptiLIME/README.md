# OptiLIME: Optimal LIME Explanations

Code and experiments used for the paper [OptiLIME: Optimized LIME Explanations for Diagnostic Computer Algorithms]().

## LIME weakpoints

* LIME suffers in particular of instability, namely repeating LIME on the same unit and the same conditions is very likely to generate different explanations.
* It is not clear how to set the kernel width: high kernel width makes the explanation more global, small kernel width obtains a more local explanation, which is more adherent (high R squared metric) to the Machine Learning model.

To ease the first one, VSI and CSI Indices measure how much the explanations retrieved are stable.  
For the second, a nice geometrical intuition can be found in the paper (and the experiments in the notebook herein).

## OptiLIME

It is mandatory to set a required level of adherence (R squared value), for the local interpretable LIME model. OptiLIME uses Bayesian Optimization and finds the highest kernel width value which guarantee the level of adherence and, at the same time, achieves the best stability level (constrained to the adherence request).  
In fact, there is an increasing trade-off between stability and kernel width   

### Experiments

In the notebooks, there are experiments confirming three interesting facts about  LIME:

* How the kenrel width influences the concept of locality of the explanations and how it affects its reliability (adherence to the ML model)
* The Ridge penalty (used by default in LIME local model) is harmful: having this penalty causes LIME to retrieve erroneous linear models and decreases the adherence of the explanations
* Discovered two trade-offs:
  * smaller kernel width implies higher adherence
  * higher kernel width implies higher stability
