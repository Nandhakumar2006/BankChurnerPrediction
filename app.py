import gradio as gr
import pickle
import pandas as pd

with open("decision_tree_model.pkl", "rb") as f:
    model = pickle.load(f)

def predict_churn(CreditScore, Geography, Gender, Age, Tenure, Balance,
                  NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary):
    data = pd.DataFrame([{
        "CreditScore": CreditScore,
        "Geography": Geography,
        "Gender": Gender,
        "Age": Age,
        "Tenure": Tenure,
        "Balance": Balance,
        "NumOfProducts": NumOfProducts,
        "HasCrCard": HasCrCard,
        "IsActiveMember": IsActiveMember,
        "EstimatedSalary": EstimatedSalary
    }])
    
    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0][1]
    
    result = "❌ Customer likely to leave (Exited)" if prediction == 1 else "✅ Customer likely to stay"
    return { "Prediction": result, "Probability of Leaving": round(probability, 3) }

demo = gr.Interface(
    fn=predict_churn,
    inputs=[
        gr.Number(label="CreditScore"),
        gr.Dropdown(["France", "Spain", "Germany"], label="Geography"),
        gr.Dropdown(["Male", "Female"], label="Gender"),
        gr.Number(label="Age"),
        gr.Number(label="Tenure"),
        gr.Number(label="Balance"),
        gr.Number(label="NumOfProducts"),
        gr.Radio([0, 1], label="HasCrCard (0/1)"),
        gr.Radio([0, 1], label="IsActiveMember (0/1)"),
        gr.Number(label="EstimatedSalary")
    ],
    outputs=gr.JSON(label="Prediction Result"),
    title="Bank Customer Churn Prediction",
    description="Enter customer details to predict if they will leave the bank."
)

if __name__ == "__main__":
    demo.launch(share=False)
