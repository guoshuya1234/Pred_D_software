## Usage
The software makes the trained LGBM model for predicting diffusion coefficient into an interactive desktop application 
for the user's convenience.Users do not need to install other ancillary software,open the main_D.exe can be used.

## This software has two functions: 
1- To calculate the gas molecular diffusion coefficient in single-crystal materials.
   A single prediction result is displayed on the interface.
2- Batch computing diffusion coefficient of crystal materials
   The predicted result will be saved in Result/Batch_Predicted_D.xlsx.

## This folder includes five folders:
1- Code
     1.LGBM_code.py that has the code for the machine learning using LGBM (for more info please visit: https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRegressor.html)
     2.Prediction_D_code.py that has the code for a human-computer interactive interface software.

2- Extrapolation_data
     Example_C2H6.xlsx that is a sample file for batch prediction of material diffusivity.

3- Img 
     full_name.png and sample_file.png that are the interactive interface software required in the illustration picture. 
 
4- model
     lgbm.pt that is a trained LGBM algorithm model.

5- Result 
     The predicted result will be automatically generated and saved in Result/Batch_Predicted_D.xlsx.



