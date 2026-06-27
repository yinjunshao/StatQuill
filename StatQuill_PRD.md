StatQuill is a prediction model that allows users to enter in databases in .csv files, excel files or any type of widely used spreadsheet styled file via uploads. The system then dissects the files in different columns with the headers being the name of the variable. It measures useful statistical data and provides an output internally, which then it feeds these information to an AI which would then be used to speak on certain trends and provide additional feedback.
The statistics will be used primarily to predict the next row of data if the data is to do with time and real time statistics, via the most recent/neighboring data points. Or it can be used to predict the value for each other column by entering specific columns and leaving the ones needed to be predicted empty. The more columns filled out the more accurate the unfilled values will be.

Scope of what is being measured:
- Autoregressiven model -> next step in terms of time by looking at the previous timestamps and multiplying them by weights based on historical data/trends.
  
  - $Y_t = c + \phi_1 Y_{t-1} + \phi_2 Y_{t-2} + \dots + \epsilon_t$
    
- Linear regression -> pred. missing field by multiplying known field. (might not be applicable if there are more than one field in the dataset so it'll be multi-axis)
  
- Residual sum of squares -> measures overall model accuracy during training, calculates the total squared distance between the model's guesses and the real data so the computer knows how to adjust
  
- Prediction variance -> calculates "wiggle room", and uncertainty of new specific predictions. The further the input is from the historical average the higher the number gets.
  
- Standard error -> Turns variance into uusable units. Taking the square roots of the predictions variance gives you the standard baseline unit for mapping out your upper and lower boundaries
  
- Prediction interval -> Creates a safety net range for a standard 95% confidence rate, it tells you the exact high and low boundaries where the next line of data will likely fall.
  
- Coefficient of variation -> Normalises uncertainty for sorting. It divides the uncertainty by the prediction size, so can fairy compare and sort different categories by their stabilities
  
- Z-score -> Find the prob of beating a target. Measures how many standard deviations away from

- Matrix form linear regression: Instead of calculating one slope, it calculates a weight ($\beta$) for *every single column* in your table simultaneously. To get the prediction, it multiplies your row of inputs by the row of weights using a dot product.
  
- Covariance Matrix: This is the heart of multi-variable risk. It is a massive grid that calculates how every single column in your dataset moves in relation to *every other column*. It tells the model if Variable A and Variable B are just repeating the same information or moving completely independently.


The system that dissects the .csv file should not have limit to how many files thats allowed to be uploaded. 

Below is the flowchart of the program in Mermaid.js format:

---
config:
  layout: dagre
---
flowchart TB
    Start["START: User launches StatQuill CLI"] --> CheckConfig["Check Configuration<br>(.env / JSON)"]
    CheckConfig --> HasAPIKey{"Is API Key set?"}
    HasAPIKey -- NO --> PromptSetup["PROMPT: Recommend Setup"]
    PromptSetup --> EnterKey["Enter OpenRouter API Key"]
    EnterKey --> SelectModel["Enter OpenRouter Model Name"]
    SelectModel --> HasAPIKey
    HasAPIKey -- YES --> PromptFile["Prompt: File Path"]
    PromptFile --> UserFile[("User pastes file path<br>e.g., /data.csv")]
    UserFile --> PromptContext["Prompt: Context<br>(Optional: domain context)"]
    PromptContext --> PressEnter["USER PRESSES ENTER"]
    PressEnter --> ParsingEngine["PARSING & QUANTITATIVE MATHEMATICS ENGINE"]
    ParsingEngine --> AIEnhancement["AI ANALYSIS ENHANCEMENT ENGINE<br>Sends Raw Math Stats + Context to the OpenRouter selected model"]
    AIEnhancement --> DisplayOutput["DISPLAY TERMINAL OUTPUT<br>1. Raw numbers &amp; stats<br>2. Humanized AI commentary"]
    DisplayOutput --> PredictionLoop["PREDICTION MODE INTERACTIVE LOOP<br>User inputs known values for columns"]
    PredictionLoop --> HasTime{"Does data track Time?"}
    HasTime -- YES --> AutoRegressive["Run Autoregressive Model using neighboring points and trends"]
    HasTime -- NO --> MatrixReg["Run Multi-Axis Matrix Linear Regression<br>"]
    AutoRegressive --> Uncertainty["CALCULATE UNCERTAINTY PROFILE"]
    MatrixReg --> Uncertainty
    Uncertainty --> FinalOutput["FINAL PREDICTION OUTPUT<br>Fills target columns + sorts by lowest CV"]

     HasAPIKey:::decision
     ParsingEngine:::engine
     AIEnhancement:::engine
     DisplayOutput:::output
     HasTime:::decision
     Uncertainty:::engine
     FinalOutput:::output
    classDef engine fill:#4ade80,stroke:#166534,color:#166534
    classDef decision fill:#fbbf24,stroke:#92400e,color:#92400e
    classDef output fill:#60a5fa,stroke:#1e40af,color:#1e40af

# Parsing:
- Detect file format (CSV, XLSX, TSV, etc.) and decode with correct encoding (UTF-8 fallback).
- Infer column data types: numeric (continuous), categorical (ordinal/nominal), datetime, or string. This determines which columns can enter regression and which is the time axis.
- Identify the time column automatically (column name hints like date, timestamp, time, or by detecting ISO/datetime formats).
- Handle missing values: Decide whether to impute (mean/median), drop rows, or treat them as prediction targets.
- Normalize headers: Strip whitespace, handle duplicates, and map user context to column names.

# Mathematics:
A. Autoregressive (AR) Model - For Time-Series Data
- **Lag matrix construction**: Build $Y_{t-1}, Y_{t-2}, \dots, Y_{t-p}$ from the target column(s).
- **Ordinary Least Squares (OLS) on lags**: Solve for $\phi_1, \phi_2, \dots, \phi_p$ and intercept $c$ in:
  $$Y_t = c + \phi_1 Y_{t-1} + \phi_2 Y_{t-2} + \dots + \epsilon_t$$
- **Order selection ($p$)**: Use information criteria like AIC or BIC to avoid overfitting.
- **Stationarity check**: Verify via Augmented Dickey-Fuller (ADF) or KPSS. If non-stationary, difference the series first.

B. Multi-Axis Matrix Linear Regression - For Non-Time Data
When predicting missing fields from known fields:

- **Design matrix $X$**: Rows = observations, columns = known input variables (plus a column of 1s for the intercept).
- **Target vector(s) $Y$**: The column(s) to predict. If multiple targets exist, this becomes a multivariate regression.
- **Normal Equation or QR/SVD decomposition**:
  $$\beta = (X^T X)^{-1} X^T Y$$ *Requirement*: You must use a numerically stable solver (SVD or pseudo-inverse) because $X^T X$ may be singular if columns are correlated.
- **Dot-product prediction**: For a new input row $x_{new}$, the prediction is $\hat{y} = x_{new} \cdot \beta$.


C. Covariance Matrix
This is the core of multi-variable risk:

- Compute the **sample covariance matrix** $\Sigma$ across all numeric columns:
  $$\Sigma_{ij} = \frac{1}{n-1} \sum_{k=1}^{n} (x_{ki} - \bar{x}_i)(x_{kj} - \bar{x}_j)$$
- **Requirements**: $O(m^2 \cdot n)$ computation for $m$ columns and $n$ rows. You need to handle multicollinearity (when $\det(\Sigma) \approx 0$) by either dropping redundant columns or using regularization (Ridge/Tikhonov).

# Uncertainty and Error Profile Engine
After generating a point prediction, you must calculate the **uncertainty envelope**:

| Metric                            | Formula / Requirement                                                                                                           |
| --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| **Residual Sum of Squares (RSS)** | $RSS = \sum (y_i - \hat{y}_i)^2$ — used to compute $\sigma^2$, the estimated error variance.                                    |
| **Prediction Variance**           | $\text{Var}(\hat{y}) = x_{new}^T (X^T X)^{-1} x_{new} \cdot \sigma^2$ — grows as the input moves away from the historical mean. |
| **Standard Error (SE)**           | $SE = \sqrt{\text{Prediction Variance}}$ — converts variance back into the original units.                                      |
| **Prediction Interval (95%)**     | $\hat{y} \pm t_{\alpha/2, df} \cdot SE$ where $df = n - k - 1$. Requires the t-distribution critical value.                     |
| **Coefficient of Variation (CV)** | $CV = \frac{SE}{mean} $                                                                                                         |
| **Z-Score**                       | $Z = \frac{\text{Target} - \hat{y}}{SE}$ — measures how many standard deviations a user-defined target is from the prediction.  |


# AI Enhancement Engine (OpenRouter Integration)

The raw statistics alone are not enough; the AI layer requires:

- **Structured prompt construction**: Feed the model a JSON payload containing:
  - Column names and inferred types
  - Summary statistics (mean, std, min, max, correlation highlights)
  - AR coefficients or regression weights ($\beta$)
  - Top covariances (which variables move together)
  - Prediction results with intervals
- **Context injection**: Append the optional user-provided domain context (e.g., "this is sales data for Q3") to the system prompt.
- **API handling**: Manage OpenRouter key, model selection, retries, and token budgeting.

# Interactive Prediction Loop

During the live prediction phase:

In the CLI, tables must be visualized, and user can select cells to enter in data with headers.

- **Partial input parsing**: Accept a row where some columns are empty/null.
- **Feature alignment**: Map user-provided values to the correct $\beta$ weights and covariance entries.
- **Conditional prediction**: Only predict the missing columns using the known columns as regressors.
- **Ranking**: Sort multiple prediction candidates by **lowest CV** (most stable/least uncertain first).

