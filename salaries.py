import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import warnings

import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import warnings

# Suppressing deprecation warnings from st.cache
warnings.filterwarnings("ignore", message="st.cache is deprecated")

def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        st.error("File not found. Please make sure the file exists at the specified location.")
        return None
    
    


# Set page title and icon
st.set_page_config(page_title="Tech Job Explorer", page_icon="📊", layout="wide", initial_sidebar_state="expanded")

def explore_and_visualize(data):
    st.title('Explore and Visualize')

    # Interactive dropdown for Job Title
    selected_job_title = st.selectbox('Select Job Title:', data['Job Title'].unique())

    # Filter the dataset based on selected Job Title
    filtered_data = data[data['Job Title'] == selected_job_title]

    # Display the filtered dataset
    st.write(filtered_data)

    st.title('Visualize for Any feature vs Target variable')

    # User input for feature selection
    selected_feature = st.selectbox('Select Feature', ['Years of Experience', 'Education Level', 'Country', 'Race', 'Gender'])

    # Plot selected feature against salary
    fig = px.scatter(data, x=selected_feature, y='Salary', color=selected_feature,
                     title=f'{selected_feature} vs Salary', template='plotly_dark')
    st.plotly_chart(fig)

    st.title('Visualize for distribution')

    # Visualize salary distribution based on job title
    if st.checkbox('Show Salary Distribution by Job Title'):
        fig = px.box(data, x='Job Title', y='Salary', points='all', color='Job Title',
                     title='Salary Distribution by Job Title', template='plotly_dark')
        st.plotly_chart(fig)

    # Visualize salary distribution based on education level
    if st.checkbox('Show Salary Distribution by Education Level'):
        fig = px.box(data, x='Education Level', y='Salary', points='all', color='Education Level',
                     title='Salary Distribution by Education Level', template='plotly_dark')
        st.plotly_chart(fig)
        
        
def predict_salary(data):
    st.title('Make Your Own Salary Prediction')

    # User input for job details
    job_title = st.selectbox('Select Job Title', data['Job Title'].unique())
    if job_title.lower().startswith('junior'):
        min_experience = 0
        max_experience = 3
    elif job_title.lower().startswith('senior'):
        min_experience = 4
        max_experience = 20  # Adjust max value as needed
    else:
        min_experience = 0
        max_experience = 20

    years_experience = st.slider('Years of Experience', min_value=min_experience, max_value=max_experience, value=min_experience)
    education_level = st.selectbox('Select Education Level', data['Education Level'].unique())
    country = st.selectbox('Select Country', data['Country'].unique())
    race = st.selectbox('Select Race', data['Race'].unique())
    gender = st.selectbox('Select Gender', data['Gender'].unique())

    # Prepare the data for prediction
    X = data[['Years of Experience', 'Education Level', 'Country', 'Race', 'Gender']]
    y = data['Salary']

    # Encode categorical variables
    label_encoders = {}
    for column in ['Education Level', 'Country', 'Race', 'Gender']:
        label_encoders[column] = LabelEncoder()
        X[column] = label_encoders[column].fit_transform(X[column])

    # Filter data based on years of experience
    X_filtered = X[X['Years of Experience'] == years_experience]

    # Train a Random Forest regressor for Junior and Senior separately
    model_junior = RandomForestRegressor(random_state=42)
    model_senior = RandomForestRegressor(random_state=42)

    # Split data into Junior and Senior
    X_junior = X_filtered
    y_junior = y[X_filtered.index]
    X_senior = X[X['Years of Experience'] > years_experience]
    y_senior = y[X['Years of Experience'] > years_experience]

    # Train models
    if not X_junior.empty:
        model_junior.fit(X_junior, y_junior)
    else:
        st.error('No data available for the selected years of experience for Junior position.')

    if not X_senior.empty:
        model_senior.fit(X_senior, y_senior)
    else:
        st.error('No data available for the selected years of experience for Senior position.')

    # Predict salary based on user input
    if years_experience <= 3:
        if st.button('Predict Junior Salary'):
            if not X_junior.empty:
                # Encode user input
                input_data = pd.DataFrame({
                    'Years of Experience': [years_experience],
                    'Education Level': [education_level],
                    'Country': [country],
                    'Race': [race],
                    'Gender': [gender]
                })
                for column in ['Education Level', 'Country', 'Race', 'Gender']:
                    input_data[column] = label_encoders[column].transform(input_data[column])

                # Predict salary based on years of experience
                if X_filtered.empty:
                    st.error('No salary prediction available for the selected years of experience.')
                else:
                    predicted_salary = model_junior.predict(input_data)
                    st.success(f'Predicted Junior Salary: ${predicted_salary[0]:,.2f}')
    else:
        if st.button('Predict Senior Salary'):
            if not X_senior.empty:
                # Encode user input
                input_data = pd.DataFrame({
                    'Years of Experience': [years_experience],
                    'Education Level': [education_level],
                    'Country': [country],
                    'Race': [race],
                    'Gender': [gender]
                })
                for column in ['Education Level', 'Country', 'Race', 'Gender']:
                    input_data[column] = label_encoders[column].transform(input_data[column])

                # Predict salary based on years of experience
                predicted_salary = model_senior.predict(input_data)
                st.success(f'Predicted Senior Salary: ${predicted_salary[0]:,.2f}')


def main():
    """Main function to run the Streamlit app."""
    data = load_data('copied_data.csv')

    if data is not None:
        # Splitting the interface into two sections
        section = st.sidebar.radio('Navigation', ['Predictive Modeling', 'Explore and Visualize'])

        if section == 'Predictive Modeling':
            predict_salary(data)
        elif section == 'Explore and Visualize':
            explore_and_visualize(data)


if __name__ == '__main__':
    main()

    
    
