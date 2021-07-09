# BPA Net Revenue Model
 
This repository contains all materials for simulating the annual net revenues of the Bonneville Power Administration as well as for identifying instruments to manage their weather-related financial risk. 

The materials herein are the products of researchers at the Center on Financial Risk in Environmental Systems at UNC Chapel Hill. Dr. Simona Denaro (former post-doc) designed and validated the original net revenue model for BPA in order to characterize BPA's financial risk. Rosa Cuppari (PhD student) utilized the model to design and evaluate financial instruments to mitigate this risk, modifying the model to produce results based on various contracts. The inputs for the model include yearly firm load (demand) for BPA and hourly load for the larger BPA service territory, daily hydropower generation, non-hydropower resources (based on historical availability), annual costs and monthly rates (based on historical data only), hourly wind generation, and regional electricity prices. The starting values or the Treasury Facility and line of credit are set to 2018 values. 

Analysis relies on stochastic generation of weather data which is translated to corresponding generation, demand, and price data through the CAPOW model. For information see https://github.com/romulus97/CAPOW_PY36.git. 
