from flask import Flask, render_template

app = Flask(__name__)

# Route for the about page


@app.route('/about')
def about():
    # Symptoms of BPD
    bpd_symptoms = [
        "Intense fear of abandonment",
        "Unstable relationships",
        "Distorted self-image",
        "Impulsive and risky behavior",
        "Chronic feelings of emptiness",
        "Intense mood swings",
        "Suicidal thoughts or self-harming behavior"
    ]

    # Symptoms of OCD
    ocd_symptoms = [
        "Obsessions (recurring, intrusive thoughts)",
        "Compulsions (repetitive behaviors or mental acts)",
        "Fear of contamination or germs",
        "Symmetry and orderliness",
        "Unwanted, aggressive or taboo thoughts",
        "Excessive checking, washing, or counting"
    ]

    # Symptoms of Anxiety
    anxiety_symptoms = [
        "Excessive worry or fear",
        "Restlessness or feeling on edge",
        "Irritability",
        "Difficulty concentrating",
        "Sleep disturbances",
        "Physical symptoms (e.g. increased heart rate, sweating)",
        "Avoidance of situations that trigger anxiety"
    ]

    # Symptoms of Depression
    depression_symptoms = [
        "Persistent sadness or low mood",
        "Loss of interest in previously enjoyable activities",
        "Changes in appetite or weight",
        "Sleep disturbances",
        "Fatigue or low energy",
        "Feelings of guilt or worthlessness",
        "Difficulty concentrating or making decisions",
        "Suicidal thoughts or behaviors"
    ]

    return render_template('about.html',
                           bpd_symptoms=bpd_symptoms,
                           ocd_symptoms=ocd_symptoms,
                           anxiety_symptoms=anxiety_symptoms,
                           depression_symptoms=depression_symptoms)


if __name__ == '__main__':
    app.run(debug=True)
