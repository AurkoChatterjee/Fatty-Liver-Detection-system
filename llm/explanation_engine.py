def generate_explanation(patient_data, risk):

    age, male, weight, height, bmi, futime, chol, dbp, fib4, hdl, sbp = patient_data

    explanation = ""

    # Risk level
    if risk > 0.5:
        explanation += "The patient shows a high risk of fatty liver disease.\n\n"
    else:
        explanation += "The patient shows a low risk of fatty liver disease.\n\n"

    # BMI
    if bmi > 30:
        explanation += "- High BMI indicates obesity, a major risk factor.\n"
    elif bmi > 25:
        explanation += "- Slightly elevated BMI may contribute to risk.\n"
    else:
        explanation += "- BMI is within normal range.\n"

    # Cholesterol
    if chol > 200:
        explanation += "- High cholesterol levels detected.\n"
    else:
        explanation += "- Cholesterol levels are normal.\n"

    # HDL
    if hdl < 40:
        explanation += "- Low HDL (good cholesterol) increases risk.\n"
    else:
        explanation += "- HDL levels are healthy.\n"

    # Blood pressure
    if sbp > 140 or dbp > 90:
        explanation += "- Elevated blood pressure observed.\n"

    # Final recommendation
    explanation += "\nRecommendation:\n"
    explanation += "Maintain a balanced diet, regular exercise, and routine health checkups."

    return explanation