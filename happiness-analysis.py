import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer

# Load dataset
st.title("World Happiness Report 2023 Analysis")
st.sidebar.title("Navigation")
option = st.sidebar.selectbox("Choose Section", ["Overview", "Data Exploration", "Regression Analysis", "Conclusion"])

if option == "Overview":
    st.header("Overview of the Dataset")
    st.write("This project explores factors affecting the Happiness Score using data from the **World Happiness Report 2023**.")
    
    df = pd.read_csv("WHR2023.csv")
    
    st.write("### Dataset Description")
    st.write("""
    The dataset includes various socio-economic and well-being indicators for countries worldwide. 
    Each row represents a country with attributes such as GDP, social support, life expectancy, 
    freedom, generosity, and corruption perceptions. The primary target variable is the **Ladder Score**, 
    which measures the overall happiness level of a country's population.
    """)
    
    st.write("### Dataset Preview")
    st.dataframe(df.head())
    
    st.write("### Columns and Descriptions")
    column_descriptions = {
        'Country name': "Name of the country.",
        'Ladder score': "The happiness score based on survey responses (target variable).",
        'Logged GDP per capita': "Natural logarithm of GDP per capita, representing economic prosperity.",
        'Social support': "Extent of social support perceived by individuals in the country.",
        'Healthy life expectancy': "Average number of healthy years a person is expected to live.",
        'Freedom to make life choices': "Freedom individuals feel they have to make key life decisions.",
        'Generosity': "Level of generosity perceived in society.",
        'Perceptions of corruption': "Level of corruption perceived in government and business sectors."
    }
    for col, desc in column_descriptions.items():
        st.write(f"- **{col}**: {desc}")
    
    st.write("Explore the other sections for deeper insights.")


elif option == "Data Exploration":
    st.header("Data Exploration")
    st.write("In this section, we visualize the data distributions and analyze correlations between variables to understand relationships and patterns.")
    
    df = pd.read_csv("WHR2023.csv")
    
    # Cleaning missing data
    df = df.dropna()  
    
    # Selecting relevant columns for correlation
    relevant_columns = ['Ladder score', 'Logged GDP per capita', 'Social support', 
                         'Healthy life expectancy', 'Freedom to make life choices', 
                         'Generosity', 'Perceptions of corruption']
    corr = df[relevant_columns].corr()
    
    # Display heatmap
    st.write("### Correlation Heatmap")
    plt.figure(figsize=(10, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", cbar=True)
    plt.title("Correlation Heatmap")
    st.pyplot(plt)
    
    st.write("""
    **Key Insights:**
    - Positive correlations (closer to +1) indicate strong positive relationships (e.g., GDP and Ladder Score).
    - Negative correlations (closer to -1) indicate inverse relationships.
    - Use this information to select predictors for the regression analysis.
    """)
    
    # Distribution of Ladder Score
    st.write("### Ladder Score Distribution")
    plt.figure(figsize=(8, 5))
    sns.histplot(df['Ladder score'], kde=True, color="green", bins=20)
    plt.title("Distribution of Ladder Score")
    plt.xlabel("Ladder Score")
    plt.ylabel("Frequency")
    st.pyplot(plt)
    
    st.write("""
    The **Ladder Score** is our target variable. Understanding its distribution helps in interpreting the model outcomes later.
    """)

elif option == "Regression Analysis":
    st.header("Regression Analysis")
    st.write("This section explores how well the selected features predict the Happiness Score (Ladder Score) using a Multiple Linear Regression model.")
    df = pd.read_csv("WHR2023.csv")

    # Cleaning missing data
    df = df.dropna()  # Drop rows with missing values

    # Selecting features and target
    X = df[['Logged GDP per capita', 'Social support', 'Healthy life expectancy', 
            'Freedom to make life choices', 'Generosity', 'Perceptions of corruption']]
    y = df['Ladder score']

    # Splitting dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Linear Regression Model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Model Coefficients Table
    coefficients = pd.DataFrame({
        "Feature": X.columns,
        "Coefficient": model.coef_,
        "Interpretation": ["For every unit increase in {} (holding other variables constant), Ladder Score changes by {:.3f}".format(feat, coef) for feat, coef in zip(X.columns, model.coef_)]
    })
    st.write("### Model Coefficients")
    st.dataframe(coefficients)
    
    # Model Performance Metrics
    st.write("### Model Performance Metrics")
    st.write(f"**R² Score:** {r2_score(y_test, y_pred):.3f}")
    st.write(f"**Mean Absolute Error (MAE):** {mean_absolute_error(y_test, y_pred):.3f}")
    st.write(f"**Mean Squared Error (MSE):** {mean_squared_error(y_test, y_pred):.3f}")

    # Actual vs Predicted Scatter Plot
    st.write("### Actual vs Predicted Scores")
    plt.figure(figsize=(8, 5))
    plt.scatter(y_test, y_pred, alpha=0.7, color='blue')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linewidth=2, label='Perfect Fit Line')
    plt.xlabel("Actual Scores")
    plt.ylabel("Predicted Scores")
    plt.title("Actual vs Predicted")
    plt.legend()
    st.pyplot(plt)

    st.write("The Actual vs Predicted plot visualizes how well the model predicts the Happiness Score. Points closer to the red line indicate better predictions, while larger deviations suggest areas for improvement. This plot helps evaluate the overall fit of the model.")


    # Residual Analysis
    st.write("### Residual Analysis")
    residuals = y_test - y_pred
    plt.figure(figsize=(8, 5))
    sns.histplot(residuals, kde=True, color='purple', bins=30)
    plt.title("Residuals Distribution")
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    st.pyplot(plt)

    st.write("Residual analysis examines the difference between the actual and predicted values. A normal distribution of residuals centered around zero indicates that the model captures the underlying patterns well. Any significant skewness or peaks suggest potential biases or missing variables.")

    # Feature Importance Visualization
    st.write("### Feature Importance")
    feature_importance = pd.DataFrame({
        "Feature": X.columns,
        "Absolute Coefficient": abs(model.coef_)
    }).sort_values(by="Absolute Coefficient", ascending=False)

    plt.figure(figsize=(8, 5))
    sns.barplot(data=feature_importance, x="Absolute Coefficient", y="Feature", palette="viridis")
    plt.title("Feature Importance")
    plt.xlabel("Absolute Coefficient Value")
    plt.ylabel("Feature")
    st.pyplot(plt)
    st.write("Feature importance highlights the relative influence of each predictor on the Happiness Score. Features with higher absolute coefficient values are more impactful, providing insights into key socio-economic factors that drive happiness.")

    # Adding individual linear regression plots
    st.write("### Linear Regression for Each Feature vs. Ladder Score")

    relevant_columns = ['Logged GDP per capita', 'Social support', 'Healthy life expectancy', 
                        'Freedom to make life choices', 'Generosity', 'Perceptions of corruption']

    feature_descriptions = {
        'Logged GDP per capita': "Economic wealth often correlates positively with happiness, as higher GDP provides access to better resources and services.",
        'Social support': "A strong social support system is often linked to better mental health and happiness levels.",
        'Healthy life expectancy': "Healthier and longer lives are directly associated with greater happiness.",
        'Freedom to make life choices': "The ability to make personal life decisions fosters a sense of autonomy and satisfaction.",
        'Generosity': "Generosity reflects a society's kindness and trust, which can enhance happiness.",
        'Perceptions of corruption': "Lower perceptions of corruption often lead to higher trust in institutions, contributing to happiness."
    }

    def plot_linear_regression(feature, target, data):
        plt.figure(figsize=(8, 5))
        sns.regplot(x=feature, y=target, data=data, ci=None, scatter_kws={'alpha':0.6}, line_kws={'color': 'red'})
        plt.title(f"Linear Regression: {feature} vs {target}")
        plt.xlabel(feature)
        plt.ylabel(target)
        st.pyplot(plt)

    # Adding dropdown for feature selection
    selected_feature = st.selectbox("Select a feature for regression analysis:", relevant_columns)
    if selected_feature:
        plot_linear_regression(selected_feature, 'Ladder score', df)
        st.write(f"The linear regression plot for **{selected_feature}** demonstrates its relationship with the Happiness Score.")
        st.write(f"**Analysis:** {feature_descriptions[selected_feature]}")

    st.write("These plots demonstrate how each socio-economic factor individually relates to the Happiness Score.")


elif option == "Conclusion":
    st.header("Conclusion and Recommendations")
    st.write("### Key Findings")
    st.write("""
    - The features with the strongest positive influence on the Happiness Score are GDP per capita, Social Support, and Healthy Life Expectancy.
    - Perceptions of Corruption and Generosity showed weaker relationships with the Ladder Score.
    - The model explains a significant portion of the variability in happiness levels, as indicated by the R² score.
    """)
    st.write("### Recommendations")
    st.write("""
    - Policymakers should focus on improving GDP, health systems, and community support to boost happiness.
    - Efforts to combat corruption may indirectly influence happiness levels.
    - Further exploration with more detailed data and non-linear models may enhance predictive accuracy.
    """)
