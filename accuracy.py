import pandas as pd

predictions_csv = "ensemble.csv"
correct_csv = "correct-testing.csv"

predictions = pd.read_csv(predictions_csv)
correct = pd.read_csv(correct_csv)

predictions = predictions.sort_values(by='filename')
correct = correct.sort_values(by='filename')

num_correct = 0
total = 0

for i in range(len(predictions.index)):
    if predictions.at[i, 'label'] == correct.at[i, 'label']:
        num_correct = num_correct + 1
    total = total + 1

accuracy = num_correct / total
print("Total Correct: " + str(num_correct))
print("Total Incorrect: " + str((total - num_correct)))
print("Accuracy " + str(accuracy))