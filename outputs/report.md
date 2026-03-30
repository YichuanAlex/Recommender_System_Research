# Mediation and Prediction Report

## Methods

Univariable and multivariable logistic regression analyses were performed to assess the independent association of A1 with FIV, adjusting for demographic and clinical covariates excluding postAS.
Multivariable logistic regression was conducted to identify independent predictors of HT, incorporating A1, FIV, postAS, and potential confounders.
Mediation analysis was implemented using a counterfactual imputation approach with a binary mediator (FIV) and binary outcome (HT); a complementary Sobel product-of-coefficients mediation with a continuous mediator (SORT) was accompanied by ρ-parameter sensitivity analysis.
Bootstrap ROC analysis with 1000 resamples compared the discriminative performance of models with and without the mediator; AUCs with 95% confidence intervals were reported.

## Results

In adjusted analysis, A1 showed an independent association with FIV (see Table 2, Panel A).
In the full model, A1 and FIV remained significant predictors of HT, with improved discrimination when the mediator was included (Model 2 AUC 0.822 [0.793–0.896] vs Model 1 AUC 0.714 [0.679–0.802]).
The indirect effect from A1 to HT via FIV was estimated as ACME 1.1462 [0.4940–2.1650], with a proportion mediated 2.640.
Adding FIV and postAS produced the highest discrimination (Model 3 AUC 0.831 [0.808–0.906]).

## Chinese Interpretation

在对人口学与临床协变量进行调整后，A1与FIV之间的关联仍然显著，提示A1可能是FIV发生的独立影响因素。
在HT的多因素模型中，纳入FIV与postAS后模型的区分度明显提升，说明中介途径对结局预测具有实际贡献。
A1→FIV→HT的间接效应（ACME）提示FIV在其中发挥部分中介作用；若ρ需要达到较大的阈值才使ACME接近0，则说明结果对未测混杂因素较为稳健。
以SORT为连续型中介的Za×Zb分析给出一致结论，并通过ρ参数敏感性分析展示潜在混杂对间接效应的影响趋势。

## Tables and Figures

Table 1: Baseline Characteristics (saved at tables/table1_baseline.csv)
Table 2: Logistic Regression Results (Panel A and B) (tables/table2_panelA_univ_A1_to_FIV.csv, tables/table2_panelA_multiv_A1_to_FIV.csv, tables/table2_panelB_ht_predictors.csv)
Table 3: Mediation Analysis Results (tables/table3_mediation.csv)
Figure 1: Mediation Diagram (figures/figure1_mediation.png)
Figure 2: Bootstrap ROC Curves (figures/figure2_bootstrap_roc.png)
Figure 3: Sensitivity Analysis Plot (figures/figure3_rho_sensitivity.png)