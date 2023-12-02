import streamlit as st

def config():
    
        original_title = '<h1 style="font-family: serif; color:white; font-size: 20px;"></h1>'
        st.markdown(original_title, unsafe_allow_html=True)


        # Set the background image
        background_image = """
        <style>
        [data-testid="stAppViewContainer"] > .main {
            background-image: url("https://www.netlabsglobal.com/wp-content/uploads/2021/06/Sentiment-Analysis-01-1920x960.jpg");
            background-size: 100vw 100vh;  # This sets the size to cover 100% of the viewport width and height
            background-position: center;  
            background-repeat: no-repeat;
        }
        </style>
        """

        st.markdown(background_image, unsafe_allow_html=True)