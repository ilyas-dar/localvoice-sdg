# LOCAL-VOICE-SDG TRAINING SCRIPT
# IN THIS TRAINING WE HAVE INCLUDED THE FOLLOWING:
"""
1. ASL asymmetric loss: helps in multi-label classification THAT IS THE MAIN DELAINGS ON OUR PROJECT.
2. K-fold cross-validation: a special one which is STRATIFIED K-FOLD CROSS-VALIDATION, which is used 
    to ensure that each fold has the same proportion of classes as the original dataset.
"""

# MY CONFIGURATION OF MODEL
MODEL_NAME="xlm-roberta-base"
NO_OF_LABELS=17 #sdgs
THE_BATCH_SIZE=16
E_POX=10
LRATE=2e-5
MY_FOLDS=5
TEST_PARTS=0.15
SEED42=42
OUTPUT_DIR="models"
CSV_PATH="data/processed/full_dataset.csv"


SDG_NAMES = {
    1: "No Poverty",
    2: "Zero Hunger",
    3: "Good Health and Well-being",
    4: "Quality Education",
    5: "Gender Equality",
    6: "Clean Water and Sanitation",
    7: "Affordable and Clean Energy",
    8: "Decent Work and Economic Growth",
    9: "Industry, Innovation and Infrastructure",
    10: "Reduced Inequalities",
    11: "Sustainable Cities and Communities",
    12: "Responsible Consumption and Production",
    13: "Climate Action",
    14: "Life Below Water",
    15: "Life On Land",
    16: "Peace, Justice and Strong Institutions",
    17: "Partnerships for the Goals"
}
