"""
orchestrator_agent.py

Decides which agent to call based on what the current turn contains:
- an uploaded image  -> vision_agent (prediction + XAI heatmaps)
- a text question     -> reporting_agent (project Q&A)

This module knows nothing about *how* those agents work internally —
only when to call them and how to shape the result for the UI layer.
"""


from agents import vision_agent, reporting_agent


def handle_request(
    image_path: str | None = None,
    question: str | None = None,
    prediction_context: dict | None = None,
):
    """
    Route requests to the appropriate agent.

    Parameters
    ----------
    image_path:
        Path to an uploaded OCT image.

    question:
        User's text question.

    prediction_context:
        Most recent OCT prediction. This allows follow-up questions
        such as "What precautions should I take?" to be answered in
        the context of the predicted retinal condition.
    """

    if image_path:
        result = vision_agent.predict_and_explain(image_path)

        return {
            "type": "prediction",
            "data": result,
        }

    if question:
        answer = reporting_agent.ask_reporting_agent(
            question,
            prediction_context=prediction_context,
        )

        return {
            "type": "answer",
            "data": answer,
        }

    return {
        "type": "error",
        "data": "Please upload an OCT image or ask a question.",
    }