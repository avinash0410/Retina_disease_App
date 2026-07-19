import os
import tempfile

import streamlit as st
from PIL import Image

from agents import orchestrator_agent



st.set_page_config(
    page_title="Retinal OCT Assistant",
    layout="wide",
)

st.title("Retinal OCT Classification & Reporting Assistant")

st.write(
    "Upload a retinal OCT image for AI-based classification and "
    "explainability analysis."
)


if "latest_prediction" not in st.session_state:
    st.session_state.latest_prediction = None

if "last_uploaded_file" not in st.session_state:
    st.session_state.last_uploaded_file = None

if "messages" not in st.session_state:
    st.session_state.messages = []



uploaded_file = st.file_uploader(
    "Upload a retinal OCT image",
    type=["jpg", "jpeg", "png"],
)


if uploaded_file is not None:

    uploaded_image = Image.open(uploaded_file)

    st.image(
        uploaded_image,
        caption=f"Uploaded image: {uploaded_file.name}",
        width=500,
    )


    analyze_button = st.button(
        "Analyze OCT Image",
        type="primary",
    )

    if analyze_button:

        with st.spinner("Analyzing retinal OCT image..."):
            file_extension = os.path.splitext(
                uploaded_file.name
            )[1]

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=file_extension,
            ) as tmp_file:

                tmp_file.write(
                    uploaded_file.getvalue()
                )

                temp_image_path = tmp_file.name

            try:


                response = orchestrator_agent.handle_request(
                    image_path=temp_image_path
                )


                if response["type"] == "prediction":

                    st.session_state.latest_prediction = (
                        response["data"]
                    )

                    st.session_state.last_uploaded_file = (
                        uploaded_file.name
                    )

                    st.session_state.messages = []

                else:

                    st.error(
                        response.get(
                            "data",
                            "Unable to analyze image.",
                        )
                    )

            except Exception as e:

                st.error(
                    f"Error analyzing image: {str(e)}"
                )

            finally:
                if os.path.exists(temp_image_path):
                    os.remove(temp_image_path)


prediction = st.session_state.get("latest_prediction")



if prediction is not None:

    st.divider()

    st.subheader("OCT Analysis Result")


    predicted_class = prediction.get(
        "predicted_class",
        "Unknown",
    )

    confidence = prediction.get(
        "confidence",
        0,
    )

    try:
        confidence_value = float(confidence)

        if confidence_value <= 1:
            confidence_display = f"{confidence_value * 100:.2f}%"
        else:
            confidence_display = f"{confidence_value:.2f}%"

    except (TypeError, ValueError):
        confidence_display = str(confidence)

    st.markdown(
        f"### Prediction: {predicted_class}"
    )

    st.write(
        f"**Confidence:** {confidence_display}"
    )


    st.subheader("Explainable AI Analysis")

    original_image = prediction.get("original_image")

    xai_images = prediction.get("images") or {}

    xai_columns = st.columns(4)


    if original_image is not None:

        with xai_columns[0]:

            st.write("**Original**")

            st.image(
                original_image,
                use_container_width=True,
            )


    gradcam = xai_images.get(
        "gradcam_overlay"
    )

    if gradcam is not None:

        with xai_columns[1]:

            st.write("**Grad-CAM**")

            st.image(
                gradcam,
                use_container_width=True,
            )


    
    gradcam_pp = xai_images.get(
        "gradcam_pp_overlay"
    )

    if gradcam_pp is not None:

        with xai_columns[2]:

            st.write("**Grad-CAM++**")

            st.image(
                gradcam_pp,
                use_container_width=True,
            )


    
    eigencam = xai_images.get(
        "eigencam_overlay"
    )

    if eigencam is not None:

        with xai_columns[3]:

            st.write("**Eigen-CAM**")

            st.image(
                eigencam,
                use_container_width=True,
            )




    top3 = prediction.get(
        "top3",
        [],
    )

    if top3:

        st.subheader(
            "Top-3 Predictions"
        )

        for item in top3:

            if isinstance(item, dict):

                class_name = item.get(
                    "class",
                    item.get(
                        "class_name",
                        "Unknown",
                    ),
                )

                probability = item.get(
                    "confidence",
                    item.get(
                        "probability",
                        0,
                    ),
                )

            else:

                class_name = item[0]
                probability = item[1]

            try:

                probability = float(
                    probability
                )

                if probability <= 1:
                    probability *= 100

                st.write(
                    f"**{class_name}:** "
                    f"{probability:.2f}%"
                )

            except Exception:

                st.write(
                    f"**{class_name}:** "
                    f"{probability}"
                )


    

    region = prediction.get(
        "region",
        "Unknown",
    )

    localization = prediction.get(
        "localization",
        "",
    )

    activation_pct = prediction.get(
        "activation_pct",
        0,
    )

    st.write(
        f"**Affected region:** {region}"
    )

    if localization:

        st.write(
            f"**Localization:** {localization}"
        )

    try:

        st.write(
            f"**Activation:** "
            f"{float(activation_pct):.1f}% of image"
        )

    except Exception:

        pass


    

    clinical_note = prediction.get(
        "clinical_note"
    )

    if clinical_note:

        st.info(
            f"**Clinical note:** {clinical_note}"
        )


    

    st.divider()

    st.subheader(
        "Ask About This Result"
    )


   
    for message in st.session_state.messages:

        with st.chat_message(
            message["role"]
        ):

            st.markdown(
                message["content"]
            )


    
    user_question = st.chat_input(
        "Ask a question about this OCT result..."
    )


    if user_question:

       
        st.session_state.messages.append(
            {
                "role": "user",
                "content": user_question,
            }
        )


        with st.chat_message("user"):

            st.markdown(
                user_question
            )


      
        try:

            with st.spinner(
                "Generating answer..."
            ):

                response = (
                    orchestrator_agent.handle_request(
                        question=user_question,
                        prediction_context=prediction,
                    )
                )


            if response["type"] == "answer":

                answer = response["data"]

            else:

                answer = response.get(
                    "data",
                    "Unable to answer the question.",
                )


        except Exception as e:

            answer = (
                f"Error generating answer: {str(e)}"
            )


        
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
            }
        )


        
        with st.chat_message(
            "assistant"
        ):

            st.markdown(
                answer
            )




else:

    st.info(
        "Upload an OCT image and click "
        "'Analyze OCT Image' to begin."
    )